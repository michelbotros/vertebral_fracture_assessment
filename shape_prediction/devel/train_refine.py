import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from load_data import load_masks, DatasetMask, split_train_val_test
from unet import UNet, Discriminator
from config import *
from torchsummary import summary
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score
import shutil


def train(n_epochs, batch_size, lr, val_percent, k, c):

    # first copy the code the experiment dir
    print('Saving experiment at: {}'.format(run_dir))
    code_dest = os.path.join(experiments_dir, run_name, 'src')
    os.makedirs(code_dest, exist_ok=True)
    shutil.copy2('config.py', code_dest)
    shutil.copy2('load_data.py', code_dest)
    shutil.copy2('train_refine.py', code_dest)
    shutil.copy2('unet.py', code_dest)

    # declare device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pre-trained coarse net, put in eval mode
    print('Loading coarse net from: {}'.format(coarse_model_path))
    coarse_net = UNet().to(device)
    coarse_net.load_state_dict(torch.load(coarse_model_path))
    coarse_net.eval()

    # construct refine net and discriminator
    refine_net = UNet(in_channels=5).to(device)
    discriminator = Discriminator().to(device)

    # load data
    xvertseg_masks, xvertseg_scores = load_masks(xvertseg_dir)
    verse2019_masks, verse2019_scores = load_masks(verse2019_dir)

    # stack data sets together
    masks = np.concatenate((xvertseg_masks, verse2019_masks))
    scores = xvertseg_scores.append(verse2019_scores, ignore_index=True)

    # make train/val split: only loads healthy in the train set
    train_set, val_set, test_set = split_train_val_test(masks, scores, patch_size, val_percent=val_percent)

    # initialize data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16, shuffle=True)

    # define loss and optimizers
    bce_logits_loss = nn.BCEWithLogitsLoss()
    optimizer_g = optim.RMSprop(refine_net.parameters(), lr=lr)
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=lr)

    # logging with wandb
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb.init(project="Shape Prediction Refinement")
    wandb.run.name = run_name

    # keep track of best validation score, for saving best model
    min_val = float('inf')

    for epoch in range(n_epochs):

        # keep track of stats discriminator this epoch
        d_train_loss = []
        d_train_acc = []

        # keep track of stats generator
        g_train_r_loss = []
        g_train_a_loss = []
        g_val_r_loss = []
        g_val_a_loss = []

        for i, (x, y, g) in enumerate(train_loader):

            if i % k != 0:

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # put batch on device
                x = x.to(device)
                y = y.to(device)

                # get a batch of fake/generated from pre-trained coarse net
                f_coarse = coarse_net.forward(x)

                # put the coarse result in context
                refine_input = torch.cat((x, f_coarse), dim=1)
                f_fine = refine_net.forward(refine_input)
                fine_hard = (f_fine > 0).to(torch.float32)

                # add context
                real = torch.cat((x, y), dim=1)
                fake = torch.cat((x, fine_hard), dim=1)

                # clear gradients
                optimizer_d.zero_grad()

                # insert real and fake vertebrae in discriminator
                real_validity = discriminator.forward(real)
                fake_validity = discriminator.forward(fake)

                # compute discriminator statistics
                y_true = torch.cat((torch.ones_like(real_validity), torch.zeros_like(fake_validity))).cpu()
                y_pred = torch.cat((real_validity > 0, fake_validity > 0)).cpu()
                d_train_acc.append(accuracy_score(y_true, y_pred))

                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                d_train_loss.append(d_loss.item())
                d_loss.backward()
                optimizer_d.step()

                # clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-c, c)

            else:

                # -----------------
                #  Train Generator
                # -----------------

                # put batch on device
                x = x.to(device)
                y = y.to(device)

                # get a batch of fake/generated from pre-trained coarse net
                f_coarse = coarse_net.forward(x)

                # put the coarse result in context
                refine_input = torch.cat((x, f_coarse), dim=1)
                f_fine = refine_net.forward(refine_input)
                fine_hard = (f_fine > 0).to(torch.float32)

                # add context (Global Discriminator)
                fake = torch.cat((x, fine_hard), dim=1)

                # clear gradients
                optimizer_g.zero_grad()

                # compute reconstruction loss
                r_loss = bce_logits_loss(f_fine, y)
                g_train_r_loss.append(r_loss.item())

                # compute adversarial loss => generator has to fool discriminator
                a_loss = -torch.mean(discriminator.forward(fake))
                g_train_a_loss.append(a_loss.item())

                # combine recon loss and adversarial loss
                g_loss = r_loss + a_loss
                g_loss.backward()
                optimizer_g.step()

        with torch.no_grad():
            for x, y, g in val_loader:

                # put batch on device
                x = x.to(device)
                y = y.to(device)

                # get a batch of fake/generated from pre-trained coarse net
                f_coarse = coarse_net.forward(x)

                # put the coarse result in context
                refine_input = torch.cat((x, f_coarse), dim=1)
                f_fine = refine_net.forward(refine_input)
                fine_hard = (f_fine.detach() > 0).to(torch.float32)

                # add context
                fake = torch.cat((x, fine_hard), dim=1)

                # compute reconstruction loss
                r_loss = bce_logits_loss(f_fine, y)
                g_val_r_loss.append(r_loss.item())

                # compute adversarial loss => generator has to fool discriminator
                fake_validity = discriminator.forward(fake)
                a_loss = -torch.mean(fake_validity)
                g_val_a_loss.append(a_loss.item())

        # avg stats discriminator
        avg_d_train_loss = np.mean(d_train_loss)
        avg_d_train_acc = np.mean(d_train_acc)

        # avg stats generator
        avg_g_train_r_loss = np.mean(g_train_r_loss)
        avg_g_train_a_loss = np.mean(g_train_a_loss)
        avg_g_val_r_loss = np.mean(g_val_r_loss)
        avg_g_val_a_loss = np.mean(g_val_a_loss)

        # print and log statistics
        print('Epoch {}\n'
              'Training: Loss D: {:.4f}, Acc D: {:.4f}, Loss G Recon: {:.4f}, Loss G Adversarial: {:.4f}\n'
              'Validation: Loss G Recon: {:.4f}, Loss G Adversarial: {:.4f}'.format(
                epoch,
                avg_d_train_loss, avg_d_train_acc, avg_g_train_r_loss, avg_g_train_a_loss,
                avg_g_val_r_loss, avg_g_val_a_loss))

        wandb.log({'d train loss': avg_d_train_loss, 'd train acc': avg_d_train_acc,
                   'g train r loss': avg_g_train_r_loss, 'g train a loss': avg_g_train_a_loss, 'g val r loss': avg_g_val_r_loss, 'g val a loss': avg_g_val_a_loss})

        # save model with best reconstruction loss
        if avg_g_val_r_loss < min_val:
            torch.save(refine_net.state_dict(),
                       os.path.join(run_dir, 'best_model_epoch_{}_loss_{:.3f}.pt'.format(epoch, avg_g_val_r_loss)))
            min_val = avg_g_val_r_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--val_percent", type=float, default=0.1, help="percentage to use for validation")
    parser.add_argument("--k", type=int, default=5, help="steps of D per step of G")
    parser.add_argument("--c", type=float, default=0.01, help="lower and upper clip value for D weights")
    args = parser.parse_args()
    train(args.n_epochs, args.batch_size, args.lr, args.val_percent, args.k, args.c)