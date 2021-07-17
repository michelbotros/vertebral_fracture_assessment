from config import *
from load_data import load_data, split_train_val_test, Dataset
from models.pl_base import Net
import torch
from torch.utils.data import DataLoader
import os
from torchsummary import summary
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import argparse
import pickle


def train(model, train_set, val_set):

    # initialize data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16, drop_last=True, shuffle=True)

    # log everything
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb_logger = WandbLogger(project="Vertebral Fracture Classification", name=run_name, save_dir=experiments_dir,
                               settings=wandb.Settings(start_method='fork'))
    wandb_logger.log_hyperparams({
        "batch_size": batch_size,
        "patch_size": patch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "data_aug": data_aug,
        "weight decay": weight_decay,
        "description": description,
        "dataset": "xVertSeg, Verse2019",
    })

    # define checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(experiments_dir, run_name),
                                          filename='{epoch:02d}_{step:03d}_{val loss grade:.2f}',
                                          monitor='val loss grade', mode='min', save_top_k=5)
    # define trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        log_every_n_steps=25,
        gpus=1,
        max_epochs=epochs,
        progress_bar_refresh_rate=0
    )

    # train the model
    trainer.fit(model, train_loader, val_loader)
    return trainer


def main(train_mode, test_mode):
    # get the model
    model = Net(lr=lr, weight_decay=weight_decay)

    # for printing the summary
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    summary(model.to(device), input_size=(2, *patch_size), batch_size=batch_size)

    # train then test
    if train_mode:
        # load data
        xvertseg_imgs, xvertseg_msks, xvertseg_scores = load_data(xvertseg_dir)
        verse2019_imgs, verse2019_msks, verse2019_scores = load_data(verse2019_dir)

        # stack data sets together
        imgs = np.concatenate((xvertseg_imgs, verse2019_imgs))
        msks = np.concatenate((xvertseg_msks, verse2019_msks))
        scores = xvertseg_scores.append(verse2019_scores)

        # split in train/val/test
        train_set, val_set, test_set = split_train_val_test(imgs, msks, scores, patch_size, data_aug)
        trainer = train(model, train_set, val_set)

        # use the best model just trained
        if test_mode:
            test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=16, shuffle=False, drop_last=True)
            print('Testing model: {}'.format(trainer.checkpoint_callback.best_model_path))

            # get results on the test set
            results = trainer.test(test_dataloaders=test_loader, ckpt_path='best')

            # save predictions
            pred_df = pd.DataFrame({'g_hat': results[0]['test g_hat'], 'c_hat': results[0]['test c_hat']})
            pred_df.to_csv(os.path.join(experiments_dir, run_name, 'preds.csv'))

            # save test set as well
            with open(os.path.join(experiments_dir, run_name, 'test_set'), 'wb') as f:
                pickle.dump(test_set, f)
    else:
        model_name = 'epoch=55_step=13103_val loss grade=0.37.ckpt'
        model_path = os.path.join(experiments_dir, run_name, model_name)
        print('Loading model from {}'.format(model_path))
        model = model.load_from_checkpoint(model_path, lr=lr, weight_decay=weight_decay)

        print('Testing on data from {}'.format(nlst_dir))
        nlst_imgs, nlst_msks, nlst_scores = load_data(nlst_dir, cases=110)
        nlst_scores = nlst_scores.to_numpy()
        print('Extracting patches...')
        test_set = Dataset(nlst_scores, nlst_imgs, nlst_msks, patch_size, transforms=False)
        print('Test set has {} vertebrae.'.format(test_set.__len__()))
        test_loader = DataLoader(test_set, batch_size=1, num_workers=16, shuffle=False, drop_last=False)

        # get IDS and vertebra label of the samples in the test set
        ids = test_set.IDS
        verts = [v + 7 for v in test_set.vertebrae]

        # log everything
        os.environ["WANDB_API_KEY"] = wandb_key
        wandb_logger = WandbLogger(project="Vertebral Fracture Classification", name=run_name + '_nlst_test')

        # make trainer and test model
        trainer = Trainer(gpus=1, logger=wandb_logger)
        results = trainer.test(model, test_dataloaders=test_loader)

        # save predictions
        pred_df = pd.DataFrame({'ID': ids, 'vert': verts, 'grade': results[0]['test g_hat']})
        pred_df.to_csv(os.path.join(experiments_dir, run_name, 'nlst_grades.csv'), index=False)

        # save test set as well
        # with open(os.path.join(experiments_dir, run_name, 'nlst_test_set'), 'wb') as f:
        #     pickle.dump(test_set, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and testing pipeline for Vertebrae Fracture Detection.')
    parser.add_argument('--train', help='run in test mode', default=False)
    parser.add_argument('--test', help='run in test mode', default=False, action='store_true')
    args = parser.parse_args()
    main(args.train, args.test)
