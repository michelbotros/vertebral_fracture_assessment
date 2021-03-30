import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
import torch
from wandb.sklearn import plot_confusion_matrix
from sklearn.metrics import accuracy_score, cohen_kappa_score
import torch.nn.functional as F


class CNN(pl.LightningModule):
    """
    CNN model for two sub-tasks: predicting grades and cases
    """
    def __init__(self, lr, weight_decay, weights_grades, weights_cases):
        super(CNN, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        # feature extraction
        self.conv1 = self.conv_block(2, 32)
        self.conv2 = self.conv_block(32, 64)
        self.conv3 = self.conv_block(64, 128)
        self.fc1 = nn.Linear(128 * 11 * 11 * 11, 256)

        # classification heads for grades and cases
        self.fc_g = nn.Linear(256, 4)
        self.fc_c = nn.Linear(256, 4)

        # weighted CCE
        self.loss_g = nn.CrossEntropyLoss(weight=torch.Tensor(weights_grades, device=self.device))
        self.loss_c = nn.CrossEntropyLoss(weight=torch.Tensor(weights_cases, device=self.device))

    def conv_block(self, in_channels, out_channels):
        conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(5, 5, 5)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d((3, 3, 3), stride=2)
        )
        return conv_block

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 11 * 11 * 11)
        x = self.fc1(x)
        g = self.fc_g(x)
        c = self.fc_c(x)
        return g, c

    def accumulate_outputs(self, outputs):
        g = torch.hstack([output['g'] for output in outputs])
        c = torch.hstack([output['c'] for output in outputs])
        g_hat = torch.hstack([output['g_hat'] for output in outputs])
        c_hat = torch.hstack([output['c_hat'] for output in outputs])
        return g.int().cpu().numpy(), c.int().cpu().numpy(), g_hat.int().cpu().numpy(), c_hat.int().cpu().numpy()

    def validation_epoch_end(self, outputs):
        g, c, g_hat, c_hat = self.accumulate_outputs(outputs)
        print('Evaluating on {} samples.'.format(len(g)))
        acc_g = accuracy_score(g, g_hat)
        acc_c = accuracy_score(c, c_hat)
        kappa_g = cohen_kappa_score(g, g_hat, weights='quadratic')
        kappa_c = cohen_kappa_score(c, c_hat, weights='quadratic')
        print('val acc_grade: {:05.2f}, val acc_case: {:.2f}, val kappa_grade: {:.2f}, val kappa_case: {:.2f}'.format(acc_g, acc_c, kappa_g, kappa_c))
        self.log_dict({'val acc_g': acc_g, 'val acc_c': acc_c, 'val kappa_c': kappa_c, 'val kappa_g': kappa_g})

    def test_epoch_end(self, outputs):
        g, c, g_hat, c_hat = self.accumulate_outputs(outputs)
        print('Testing on {} samples.'.format(len(g)))
        acc_g = accuracy_score(g, g_hat)
        acc_c = accuracy_score(c, c_hat)
        kappa_g = cohen_kappa_score(g, g_hat, weights='quadratic')
        kappa_c = cohen_kappa_score(c, c_hat, weights='quadratic')
        plot_confusion_matrix(g, g_hat, labels=['Healthy', 'Mild', 'Moderate', 'Severe'])
        self.log_dict({'test g_hat': g_hat, 'test c_hat': c_hat,
                       'test acc_g': acc_g, 'test acc_c': acc_c,
                       'test kappa_c': kappa_c, 'test kappa_g': kappa_g})

    def training_step(self, batch, batch_idx):
        x, g, c = batch
        logits_g, logits_c = self.forward(x)
        loss_g = self.loss_g(logits_g, g)
        loss_c = self.loss_c(logits_c, c)
        loss = loss_g + loss_c

        g_hat = np.argmax(F.log_softmax(logits_g, dim=1).detach().cpu(), axis=1)
        c_hat = np.argmax(F.log_softmax(logits_c, dim=1).detach().cpu(), axis=1)
        acc_g = accuracy_score(g.int().cpu(), g_hat)
        acc_c = accuracy_score(c.int().cpu(), c_hat)
        print('train loss grade: {:05.2f}, loss case: {:05.2f}, loss total: {:05.2f}, acc grade: {:05.2f}, acc case: {:05.2f}'.format(loss_g, loss_c, loss, acc_g, acc_c))
        self.log_dict({'train loss': loss.item(), 'train acc grade': acc_g, 'train acc case': acc_c}, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, g, c = batch
        logits_g, logits_c = self.forward(x)
        loss_g = self.loss_g(logits_g, g)
        loss_c = self.loss_c(logits_c, c)
        loss = loss_g + loss_c
        g_hat = np.argmax(F.log_softmax(logits_g, dim=1).cpu(), axis=1)
        c_hat = np.argmax(F.log_softmax(logits_c, dim=1).cpu(), axis=1)
        self.log_dict({'val loss': loss.item()}, on_epoch=True, on_step=False)
        return {'g': g, 'c': c, 'g_hat': g_hat, 'c_hat': c_hat}

    def test_step(self, batch, batch_idx):
        x, g, c = batch
        logits_g, logits_c = self.forward(x)
        g_hat = np.argmax(F.log_softmax(logits_g, dim=1).cpu(), axis=1)
        c_hat = np.argmax(F.log_softmax(logits_c, dim=1).cpu(), axis=1)
        return {'g': g, 'c': c, 'g_hat': g_hat, 'c_hat': c_hat}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
