import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
import torch
from wandb.sklearn import plot_confusion_matrix, plot_roc
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from pytorch_lightning.metrics.functional import accuracy


class CNN(pl.LightningModule):
    """
    An adjustable 3D CNN that performs binary classification: fractured or not.
    """
    def __init__(self, lr, groups, batch_norm, n_linear, init_filters, dropout):
        super(CNN, self).__init__()

        # net configs
        self.lr = lr
        self.groups = groups
        self.batch_norm = batch_norm
        self.n_linear = n_linear
        self.init_filters = init_filters
        self.dropout = nn.Dropout(p=0.25) if dropout else None

        # the net
        self.conv1 = self.conv_block(2, self.init_filters)
        self.conv2 = self.conv_block(self.init_filters, self.init_filters * 2)
        self.conv3 = self.conv_block(self.init_filters * 2, self.init_filters * 4)
        self.conv4 = self.conv_block(self.init_filters * 4, self.init_filters * 8)
        self.conv5 = self.conv_block(self.init_filters * 8, self.init_filters * 16)
        self.fc1 = nn.Linear(self.init_filters * 16 * 2 * 2 * 2, self.n_linear)
        self.fc2 = nn.Linear(self.n_linear, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), groups=self.groups),
                  nn.ReLU()]

        if self.batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))

        layers.append(nn.MaxPool3d((2, 2, 2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, self.init_filters * 16 * 2 * 2 * 2)
        if self.dropout:
            x = self.dropout(x)
        x = self.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

    def loss(self, logits, y):
        bce = nn.BCEWithLogitsLoss()
        return bce(logits, y)

    def accumulate_outputs(self, outputs):
        y = torch.stack([output['y'] for output in outputs]).view((-1, 1))
        logits = torch.stack([output['logits'] for output in outputs]).view((-1, 1))
        y_hat = torch.gt(logits, 0)
        y_prob = self.sigmoid(logits)
        loss = self.loss(logits, y)
        return y.int().cpu().numpy(), y_hat.int().cpu().numpy(), y_prob.cpu().numpy(), loss.cpu().numpy()

    def validation_epoch_end(self, outputs):
        y, y_hat, y_prob, loss = self.accumulate_outputs(outputs)
        print('Evaluating on {} samples.'.format(len(y)))
        acc = accuracy_score(y, y_hat)
        auc = roc_auc_score(y, y_prob)
        f1 = f1_score(y, y_hat)
        print('val loss: {:05.2f}, val acc: {:.2f}, val f1: {:.2f}, val auroc: {:.2f}'.format(loss, acc, f1, auc))
        self.log_dict({'val loss': loss, 'val acc': acc, 'val f1': f1, 'val auroc': auc})

    def test_epoch_end(self, outputs):
        y, y_hat, y_prob, loss = self.accumulate_outputs(outputs)
        print('Testing on {} samples.'.format(len(y)))
        acc = accuracy_score(y, y_hat)
        auc = roc_auc_score(y, y_prob)
        f1 = f1_score(y, y_hat)

        # to plot ROC
        y_prob_exp = np.concatenate((1 - y_prob, y_prob), axis=1)
        plot_confusion_matrix(y, y_hat, labels=['Healthy', 'Fractured'])
        plot_roc(y, y_prob_exp, labels=['Healthy', 'Fractured'], classes_to_plot=[1])
        print('test loss: {:05.2f}, test acc: {:.2f}, test f1: {:.2f}, test auroc: {:.2f}'.format(loss, acc, f1, auc))
        self.log_dict({'test loss': loss, 'test acc': acc, 'test f1': f1, 'test auroc': auc})

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        y_hat = torch.gt(logits, 0)
        acc = accuracy(y, y_hat)
        print('train loss: {:05.2f}, train acc: {:.2f}'.format(loss, acc))
        self.log_dict({'train loss': loss.item(), 'train acc': acc.item()}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        return {'y': y, 'logits': logits}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        return {'y': y, 'logits': logits}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

