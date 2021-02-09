from config import data_dir, base_dir, resolution, train_val_split, patch_size, batch_size, epochs, lr, wandb_key
from load_data import load_data
from tqdm import tqdm
import numpy as np
import wandb
from models import CNN
import torch
import torch.nn as nn
from torch.optim import Adam
import os


def main():
    # set device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get train and val loader
    train_loader, val_loader = load_data(data_dir, resolution, train_val_split, patch_size, batch_size, nr_imgs=5)

    # get the model, optimizer and loss function
    model = CNN().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    # keep track of stuff with wandb
    # TODO: add information about data set (splt, frequency fractures etc..)
    os.environ["WANDB_API_KEY"] = wandb_key

    run = wandb.init(project="binary_classifier_xVertSeg",
                     config={
                         "batch_size": batch_size,
                         "patch_size": patch_size,
                         "learning_rate": lr,
                         "epochs": epochs,
                         "optimizer": optimizer,
                         "dataset": "xVertSeg",
                     })

    # run the training loop
    for epoch in tqdm(range(epochs)):

        # training
        train_loss = []
        train_acc = []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # make predictions
            y_pred = model(X)

            # compute train_loss and backward
            loss = bce(y_pred, y)
            loss.backward()
            optimizer.step()

            # compute train acc
            y_pred = torch.round(y_pred)
            acc = torch.mean((y == y_pred).to(dtype=torch.float32))

            # store the info for this train batch
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        # validation
        val_loss = []
        val_acc = []

        for X, y in val_loader:
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                y_pred = model(X)

                loss = bce(y_pred, y)
                y_pred = torch.round(y_pred)
                acc = torch.mean((y == y_pred).to(dtype=torch.float32))

            # store the info for this val batch
            val_loss.append(loss.item())
            val_acc.append(acc.item())

        # logging
        wandb.log({'train loss': np.mean(train_loss), 'train acc': np.mean(train_acc), 'val loss': np.mean(val_loss),
                   'val acc': np.mean(val_acc)})
    run.finish()


if __name__ == '__main__':
    # add arguments and such
    main()
