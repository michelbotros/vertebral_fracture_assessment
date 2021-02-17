from config import xvertseg_dir, verse2019_dir, resolution, train_val_split, patch_size, batch_size, epochs, lr, \
    wandb_key
from load_data import load_data, split_train_val, Sampler
from models import CNN
import torch
from torch.utils.data import DataLoader
import os
from torchsummary import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import numpy as np


def main():
    # set device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data from corresponding data dir
    xvertseg_imgs, xvertseg_msks, xvertseg_scores = load_data(xvertseg_dir, resolution)
    verse2019_imgs, verse2019_msks, verse2019_scores = load_data(verse2019_dir, resolution)

    # stack data sets together
    imgs = np.concatenate((xvertseg_imgs, verse2019_imgs))
    msks = np.concatenate((xvertseg_msks, verse2019_msks))
    scores = xvertseg_scores.append(verse2019_scores)

    # split in train/val/test
    train_set, val_set, train_IDs, val_IDs = split_train_val(imgs, msks, scores, train_val_split, patch_size)

    # initialize data loaders, use custom sampling that ensures one positive sample per batch
    train_loader = DataLoader(train_set, batch_sampler=Sampler(train_set.scores, batch_size), num_workers=8)
    val_loader = DataLoader(val_set, batch_sampler=Sampler(val_set.scores, batch_size), num_workers=8)

    # get the model and put on device
    model = CNN().to(device)
    summary(model, input_size=(2, *patch_size), batch_size=batch_size)

    # log everything
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb_logger = WandbLogger(project="binary_classifier_xVertSeg")
    wandb_logger.log_hyperparams({
                         "batch_size": batch_size,
                         "patch_size": patch_size,
                         "learning_rate": lr,
                         "epochs": epochs,
                         "dataset": "xVertSeg",
                         "train_ids": train_IDs,
                         "val_ids": val_IDs
                     })

    # define trainer
    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        gpus=-1,
        max_epochs=epochs,
        deterministic=True
    )

    # train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
