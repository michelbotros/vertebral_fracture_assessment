import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import torch
from pytorch_lightning.metrics.functional import accuracy, auroc


class CNN(pl.LightningModule):
    """
    Initial 3D CNN.
    Three blocks of: Conv => Relu => (Batch Norm) => MaxPool.
    Two fully connected layers.
    """
    def __init__(self, lr, groups, batch_norm):
        super(CNN, self).__init__()

        # net configs
        self.lr = lr
        self.groups = groups
        self.batch_norm = batch_norm

        # the net
        self.conv1 = self.conv_block(2, 32)
        self.conv2 = self.conv_block(32, 64)
        self.conv3 = self.conv_block(64, 128)
        self.fc1 = nn.Linear(128 * 14 * 14 * 14, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
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
        x = x.view(-1, 128 * 14 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def loss(self, logits, y):
        bce = nn.BCEWithLogitsLoss()
        return bce(logits, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        y_hat = torch.gt(logits, 0).int()

        d = {'train loss': loss.item(), 'train acc': accuracy(y, y_hat).item()}
        print('train loss: {:05.2f}, train acc: {:.2f}'.format(d['train loss'], d['train acc']))
        self.log_dict(d)
        return loss

    def validation_epoch_end(self, outputs):
        # accumulate labels and logits
        y = torch.stack([output['y'] for output in outputs]).view(-1)
        logits = torch.stack([output['logits'] for output in outputs]).view(-1)

        # compute hard predictions, probabilities and loss
        y_hat = torch.gt(logits, 0).int()
        y_prob = self.sigmoid(logits)
        loss = self.loss(logits, y)

        print('Evaluating on {} samples.'.format(len(y)))
        d = {'val loss': loss.item(),
             'val acc': accuracy(y, y_hat).item(),
             'val auroc': auroc(y_prob, y.int(), pos_label=1).item()}

        print('val loss: {:05.2f}, val acc: {:.2f}, val auroc: {:.2f}'.format(d['val loss'],
                                                                              d['val acc'],
                                                                              d['val auroc']
                                                                              ))
        self.log_dict(d)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        return {'y': y, 'logits': logits}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

