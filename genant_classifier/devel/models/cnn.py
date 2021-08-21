import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    """
    CNN model for two sub-tasks: predicting grades and cases
    """
    def __init__(self):
        super(CNN, self).__init__()

        # feature extraction
        self.conv1 = self.conv_block(2, 32)
        self.conv2 = self.conv_block(32, 64)
        self.conv3 = self.conv_block(64, 128)
        self.conv4 = self.conv_block(128, 256)
        self.conv5 = self.conv_block(256, 512)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.maxpool = nn.MaxPool3d((2, 2, 2), stride=2)

        # classification heads for grades and cases
        self.fc_c = nn.Linear(512, 4)
        self.fc_g = nn.Linear(512, 4)

    def conv_block(self, in_channels, out_channels):
        conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(p=0.2),
            nn.ReLU()
        )
        return conv_block

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        if x.size(-1) > 4:
             x = self.maxpool(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        g = self.fc_g(x)
        c = self.fc_c(x)

        return g, c