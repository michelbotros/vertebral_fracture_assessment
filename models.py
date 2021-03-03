import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
import torch
from wandb.sklearn import plot_confusion_matrix, plot_roc
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from pytorch_lightning.metrics.functional import accuracy


class ResNetPl(pl.LightningModule):
    """
    ResNet model for binary classification.
    """

    def __init__(self, lr, weight_decay):
        super(ResNetPl, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.ResNet10 = generate_model(model_depth=10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.ResNet10.forward(x)

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


""""
3D ResNet from: https://github.com/kenshohara/3D-ResNets-PyTorch
Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, and Yutaka Satoh,
"Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs",
arXiv preprint, arXiv:2004.04968, 2020.
"""


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=2,
                 dropout=True,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, 7, 7),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.dropout = nn.Dropout(p=0.5) if dropout else None

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
