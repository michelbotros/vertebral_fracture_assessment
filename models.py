import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from config import lr


class CNN(pl.LightningModule):
    """
    Initial (simple) 3D CNN, that performs binary classification with as input a 3D patch of mask of vertebrae.
    As vanilla as possible: Conv => ReLU => MaxPool and two fully connected layers with a Sigmoid at the end.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = self.conv_block(1, 16)
        self.conv2 = self.conv_block(16, 32)
        self.conv3 = self.conv_block(32, 64)
        self.fc1 = nn.Linear(64 * 14 * 14 * 14, 32)
        self.fc2 = nn.Linear(32, 1)

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
        bce = nn.BCEWithLogitsLoss()
        loss = bce(out, y)
        self.log('train loss', loss)

        # compute metrics
        sigmoid = nn.Sigmoid()
        y_prob = sigmoid(out)
        y_pred = torch.round(y_prob)

        y = y.cpu().detach().numpy()
        y_prob = y_prob.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        self.log('train acc', accuracy_score(y, y_pred))
        self.log('train roc auc', roc_auc_score(y, y_prob))
        # self.log('train precision', precision_score(y, y_pred))
        # self.log('train sensitivity', recall_score(y, y_pred))
        # self.log('train specificity', recall_score(y, y_pred, pos_label=0))
        return loss

    def validation_step(self, batch, batch_idx):
        # forward batch
        x, y = batch
        out = self(x)

        # compute loss
        bce = nn.BCEWithLogitsLoss()
        loss = bce(out, y)
        self.log('val loss', loss)

        # compute metrics
        sigmoid = nn.Sigmoid()
        y_prob = sigmoid(out)
        y_pred = torch.round(y_prob)

        y = y.cpu().detach().numpy()
        y_prob = y_prob.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        self.log('val acc', accuracy_score(y, y_pred))
        self.log('val roc auc', roc_auc_score(y, y_prob))
        # self.log('val precision', precision_score(y, y_pred))
        # self.log('val sensitivity', recall_score(y, y_pred))
        # self.log('val specificity', recall_score(y, y_pred, pos_label=0))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=lr)
        return optimizer

