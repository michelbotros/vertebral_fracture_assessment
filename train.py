from config import data_dir, resolution, train_val_split, patch_size, batch_size, epochs, lr, wandb_key, use_weights
from load_data import load_data
from models import CNN
import torch
import os
from torchsummary import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


def main():
    # set device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get train and val loader
    train_loader, val_loader, train_ids, val_ids, weight = load_data(data_dir, resolution, train_val_split, patch_size,
                                                                     batch_size)
    # put weight tensor on device
    w = torch.tensor(weight, device=device)

    # get the model, optimizer and loss function
    model = CNN(w).to(device)
    summary(model, input_size=(1, *patch_size), batch_size=batch_size)

    # log everything
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb_logger = WandbLogger(project="binary_classifier_xVertSeg")
    wandb_logger.log_hyperparams({
                         "batch_size": batch_size,
                         "patch_size": patch_size,
                         "learning_rate": lr,
                         "weight": weight,
                         "use_weights": use_weights,
                         "epochs": epochs,
                         "dataset": "xVertSeg",
                         "train_ids": train_ids,
                         "val_ids": val_ids
                     })

    # train the model
    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        gpus=-1,
        max_epochs=epochs,
        deterministic=True
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
