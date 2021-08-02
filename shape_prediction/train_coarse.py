import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from load_data import load_masks, DatasetMask, split_train_val_test
from unet import UNet
from config import *
from torchsummary import summary
import torch.optim as optim
import wandb


def train(n_epochs, batch_size, lr, val_percent=0.1):

    # declare device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # construct UNet and print summary
    unet = UNet().to(device)

    # load data
    xvertseg_masks, xvertseg_scores = load_masks(xvertseg_dir)
    verse2019_masks, verse2019_scores = load_masks(verse2019_dir)

    # stack data sets together
    masks = np.concatenate((xvertseg_masks, verse2019_masks))
    scores = xvertseg_scores.append(verse2019_scores, ignore_index=True)

    # make train/val split: only loads healthy in the train set
    train_set, val_set, test_set = split_train_val_test(masks, scores, patch_size, val_percent=val_percent)

    print('Saving experiment at: {}'.format(run_dir))
    os.mkdir(run_dir)

    # initialize data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16, shuffle=True)

    # define loss and optimizer
    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    # logging with wandb
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb.init(project="Shape Prediction V2")
    wandb.run.name = run_name

    # keep track of best validation score, for saving best model
    min_val = float('inf')

    for epoch in range(n_epochs):

        train_loss = 0.0
        val_loss = 0.0

        for x, y, g in train_loader:  # x is 4 input channels with neighbours, y is healthy vert to be predicted

            # clear gradients
            optimizer.zero_grad()

            # forward batch
            y_pred = unet.forward(x.to(device))

            # compute loss & update
            loss = bce(y_pred, y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validate
        with torch.no_grad():
            for x, y, g in val_loader:

                # forward batch
                y_pred = unet.forward(x.to(device))

                # compute loss & update
                loss = bce(y_pred, y.to(device))
                val_loss += loss.item()

        # log & print stats
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print('Epoch {}, train loss: {:.3f}, val loss: {:.3f}'.format(epoch, avg_train_loss, avg_val_loss))
        wandb.log({'train loss': avg_train_loss, 'val loss': avg_val_loss})

        # save best
        if avg_val_loss < min_val:
            torch.save(unet.state_dict(),
                       os.path.join(run_dir, 'best_model_epoch_{}_loss_{:.3f}.pt'.format(epoch, avg_val_loss)))
            min_val = avg_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--val_percent", type=float, default=0.1, help="percentage to use for validation")
    args = parser.parse_args()
    train(args.n_epochs, args.batch_size, args.lr, args.val_percent)