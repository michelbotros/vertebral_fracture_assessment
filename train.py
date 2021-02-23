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


def main():
    """
    TODO: split this in train and test function, add arguments to run testing
    """
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
    train_set, val_set, test_set = split_train_val_test(imgs, msks, scores, patch_size, data_aug)

    # initialize data loaders, use custom sampling that ensures one positive sample per batch
    train_loader = DataLoader(train_set, batch_sampler=Sampler(train_set.get_scores(), batch_size), num_workers=8)
    val_loader = DataLoader(val_set, batch_sampler=Sampler(val_set.get_scores(), batch_size), num_workers=8)

    # get the model and put on device
    model = CNN(lr=lr, groups=groups, batch_norm=batch_norm).to(device)
    summary(model, input_size=(2, *patch_size), batch_size=batch_size)

    # log everything
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb_logger = WandbLogger(project="binary_classifier", name=run_name, save_dir=experiments_dir)
    wandb_logger.log_hyperparams({
                         "batch_size": batch_size,
                         "patch_size": patch_size,
                         "learning_rate": lr,
                         "epochs": epochs,
                         "batch_norm": batch_norm,
                         "group_conv:": groups,
                         "data_aug:": data_aug,
                         "dataset": "xVertSeg, Verse2019",
                     })

    # define checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(experiments_dir, run_name),
                                          filename='{epoch:02d}_{step:03d}_{val auroc:.2f}_{val acc:.2f}',
                                          monitor='val auroc', mode='max', save_top_k=5)
    # define trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        log_every_n_steps=1,
        val_check_interval=25,
        limit_val_batches=0.5,
        gpus=1,
        max_epochs=epochs,
        progress_bar_refresh_rate=0
    )

    # train the model
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

    # TODO: only at the end add testing!
    # model = CNN.load_from_checkpoint(checkpoint_path=)
    # test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8)
    # trainer.test(model)


if __name__ == '__main__':
    main()
