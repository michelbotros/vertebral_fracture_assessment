from config import *
from load_data import load_data, split_train_val_test, Sampler
from models import CNN
import torch
from torch.utils.data import DataLoader
import os
from torchsummary import summary
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import argparse


def train(model, train_set, val_set):
    # initialize data loaders, use custom sampling that ensures one positive sample per batch
    train_loader = DataLoader(train_set, batch_sampler=Sampler(train_set.get_scores(), batch_size), num_workers=12)
    val_loader = DataLoader(val_set, batch_sampler=Sampler(val_set.get_scores(), batch_size), num_workers=12)

    # log everything
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb_logger = WandbLogger(project="binary_classifier", name=run_name, save_dir=experiments_dir)
    wandb_logger.log_hyperparams({
        "batch_size": batch_size,
        "patch_size": patch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_norm": batch_norm,
        "group_conv": groups,
        "n_linear": n_linear,
        "init_filters": init_filters,
        "dropout": dropout,
        "data_aug:": data_aug,
        "dataset": "xVertSeg, Verse2019",
    })

    # define checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(experiments_dir, run_name),
                                          filename='{epoch:02d}_{step:03d}_{val auroc:.2f}_{val f1:.2f}{val acc:.2f}',
                                          monitor='val auroc', mode='max', save_top_k=5)
    # define trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        log_every_n_steps=1,
        val_check_interval=50,
        accelerator='dp',
        gpus=4,
        max_epochs=epochs,
        progress_bar_refresh_rate=0
    )

    # train the model
    trainer.fit(model, train_loader, val_loader)
    return trainer


def main(train_mode, test_mode):
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get the model and put on device
    model = CNN(lr, groups, batch_norm, n_linear, init_filters, dropout).to(device)
    summary(model, input_size=(2, *patch_size), batch_size=batch_size)

    # load data from corresponding data dir
    xvertseg_imgs, xvertseg_msks, xvertseg_scores = load_data(xvertseg_dir)
    verse2019_imgs, verse2019_msks, verse2019_scores = load_data(verse2019_dir)

    # stack data sets together
    imgs = np.concatenate((xvertseg_imgs, verse2019_imgs))
    msks = np.concatenate((xvertseg_msks, verse2019_msks))
    scores = xvertseg_scores.append(verse2019_scores)

    # split in train/val/test
    train_set, val_set, test_set = split_train_val_test(imgs, msks, scores, patch_size, data_aug)

    # train
    if train_mode:
        trainer = train(model, train_set, val_set)

        # use the best model just trained
        if test_mode:
            test_loader = DataLoader(test_set, batch_size=1, num_workers=12)
            results = trainer.test(test_dataloaders=test_loader, ckpt_path='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and testing pipeline for Vertebrae Fracture Detection.')
    parser.add_argument('--train', help='run in test mode', default=True)
    parser.add_argument('--test', help='run in test mode', default=True, action='store_true')
    parser.add_argument('--gpus', help='how many gpus to use', default=1)
    args = parser.parse_args()
    main(args.train, args.test)
