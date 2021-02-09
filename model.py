import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Initial (simple) 3D CNN, that performs binary classification with as input a 3D patch of mask of vertebrae.
    As vanilla as possible: Conv => ReLU => MaxPool (2x) and two fully connected layers with a Sigmoid at the end.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = self.conv_block(1, 16)
        self.conv2 = self.conv_block(16, 32)
        self.conv3 = self.conv_block(32, 64)
        self.fc1 = nn.Linear(64 * 12 * 12 * 12, 8)
        self.fc2 = nn.Linear(8, 1)
        self.final = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        return conv_block

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 12 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = self.final(x)
        return out
