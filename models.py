import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import torch
from pytorch_lightning.metrics.functional.classification import accuracy, auroc, precision, recall
from config import lr, use_weights


class CNN(pl.LightningModule):
    """
    Initial (simple) 3D CNN, that performs binary classification with as input a 3D patch of mask of vertebrae.
    As vanilla as possible: Conv => ReLU => MaxPool and two fully connected layers with a Sigmoid at the end.
    """
    def __init__(self, weight):
        super(CNN, self).__init__()
        self.conv1 = self.conv_block(1, 16)
        self.conv2 = self.conv_block(16, 32)
        self.conv3 = self.conv_block(32, 64)
        self.fc1 = nn.Linear(64 * 14 * 14 * 14, 32)
        self.fc2 = nn.Linear(32, 1)

        # class weight
        self.weight = weight

    def conv_block(self, in_channels, out_channels):
        conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        return conv_block

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 14 * 14 * 14)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out

    def training_step(self, batch, batch_idx):
        # forward batch
        x, y = batch
        out = self(x)

        # compute loss
        if use_weights:
            bce = nn.BCEWithLogitsLoss(pos_weight=self.weight)
        else:
            bce = nn.BCEWithLogitsLoss()

        loss = bce(out, y)
        self.log('train loss', loss, on_epoch=True)

        # compute metrics
        sigmoid = nn.Sigmoid()
        y_pred_soft = sigmoid(out)
        y_pred_hard = torch.round(y_pred_soft)
        self.log('train acc', accuracy(y_pred_hard, y), on_epoch=True)
        # self.log('train auroc', auroc(y_pred_soft, y), on_epoch=True)
        self.log('train precision', precision(y_pred_soft, y, num_classes=1), on_epoch=True)
        self.log('train sensitivity', recall(y_pred_soft, y, num_classes=1), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # forward batch
        x, y = batch
        out = self(x)

        # compute loss
        if use_weights:
            bce = nn.BCEWithLogitsLoss(pos_weight=self.weight)
        else:
            bce = nn.BCEWithLogitsLoss()

        loss = bce(out, y)
        self.log('val loss', loss)

        # compute metrics
        sigmoid = nn.Sigmoid()
        y_pred_soft = sigmoid(out)
        y_pred_hard = torch.round(y_pred_soft)
        self.log('val acc', accuracy(y_pred_hard, y), on_epoch=True)
        # self.log('val auroc', auroc(y_pred_soft, y), on_epoch=True)
        self.log('val precision', precision(y_pred_soft, y, num_classes=1), on_epoch=True)
        self.log('val sensitivity', recall(y_pred_soft, y, num_classes=1), on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=lr)

        return optimizer

