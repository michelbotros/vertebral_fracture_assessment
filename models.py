import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
import torch
from wandb.sklearn import plot_confusion_matrix, plot_roc
from sklearn.metrics import accuracy_score, roc_auc_score


class CNN(pl.LightningModule):
    """
    ResNet model for grading.
    """
    def __init__(self, lr, weight_decay):
        super(CNN, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x

    def loss(self, logits, y):
        bce = nn.BCEWithLogitsLoss()
        return bce(logits, y)

    def accumulate_outputs(self, outputs):
        y = torch.stack([output['y'] for output in outputs]).view((-1, 1))
        logits = torch.stack([output['logits'] for output in outputs]).view((-1, 1))

        # networks targets are [0., 0.33, 0.66, 1.]  for grades [0., 1., 2., 3.]
        y_mapped = y / 3
        loss = self.loss(logits, y_mapped)
        y_prob = self.sigmoid(logits)
        return y.int().cpu().numpy(), y_prob.cpu().numpy(), loss.cpu().numpy()

    def validation_epoch_end(self, outputs):
        y, y_prob, loss = self.accumulate_outputs(outputs)
        y_hat = np.round(y_prob * 3)         # compute hard predictions
        print('Evaluating on {} samples.'.format(len(y)))
        acc = accuracy_score(y, y_hat)
        print('val loss: {:05.2f}, val acc: {:.2f}'.format(loss, acc))
        self.log_dict({'val loss': loss, 'val acc': acc})

    def test_epoch_end(self, outputs):
        y, y_prob, loss = self.accumulate_outputs(outputs)
        print('Testing on {} samples.'.format(len(y)))
        y_hat = np.round(y_prob * 3)         # compute hard predictions
        acc = accuracy_score(y, y_hat)
        plot_confusion_matrix(y, y_hat, labels=['Healthy', 'Mild', 'Moderate', 'Severe'])
        print('test loss: {:05.2f}, test acc: {:.2f}'.format(loss, acc))
        self.log_dict({'test loss': loss, 'test acc': acc, 'y_prob': y_prob})

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        # networks targets are [0., 0.33, 0.66, 1.] for grades [0., 1., 2., 3.]
        y_mapped = y / 3
        loss = self.loss(logits, y_mapped)

        # get the hard predictions
        y_hat = torch.round(self.sigmoid(logits) * 3).int().cpu().numpy()
        y = y.int().cpu().numpy()

        # log loss and accuracy
        acc = accuracy_score(y, y_hat)
        print('train loss: {:05.2f}, train acc: {:.2f}'.format(loss, acc))
        self.log_dict({'train loss': loss.item(), 'train acc': acc.item()}, sync_dist=True, on_epoch=True, on_step=True)
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
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer