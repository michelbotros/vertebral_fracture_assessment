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
from torchsummary import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


def main():
    # set device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get train and val loader
    train_loader, val_loader = load_data(data_dir, resolution, train_val_split, patch_size, batch_size, nr_imgs=5)

    # get the model, optimizer and loss function
    model = CNN().to(device)
    summary(model, input_size=(1, *patch_size), batch_size=batch_size)

    # keep track of stuff with wandb
    # TODO: add information about data set (the model, split, frequency fractures etc..)
    os.environ["WANDB_API_KEY"] = wandb_key

    # run the training loop
    wandb_logger = WandbLogger()

    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=50,
        gpus=-1,
        max_epochs=100,
        deterministic=True
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    # add arguments and such
    main()
