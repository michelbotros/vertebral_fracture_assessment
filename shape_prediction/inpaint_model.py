import torch
from torch import nn
import numpy as np
from config import image_shape
import torch.nn.functional as F


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, rate=1, deconv=False):
        super(GatedConv, self).__init__()

        if rate == 1:
            padding = kernel_size // 2
        else:
            padding = rate

        if deconv:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=rate)
            )
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=rate)

        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)

        # the last conv => no gating
        if self.out_channels == 3:
            return x

        x, y = torch.split(x, self.out_channels // 2, dim=1)
        x = self.activation(x)
        y = self.sigmoid(y)
        x = x * y
        return x


class InPaintNet(nn.Module):
    """
    InPaintNet (Generator) Net
    input: two channel: image and mask
    output: completed mask
    TODO: (1) freeform => gated
    """

    def __init__(self, ch=3):
        super(InPaintNet, self).__init__()

        # 6x Gated Convolution
        self.conv1 = GatedConv(ch, 32, 5, 1)
        self.conv2 = GatedConv(16, 64, 3, 2)
        self.conv3 = GatedConv(32, 64, 3, 1)
        self.conv4 = GatedConv(32, 128, 3, 2)
        self.conv5 = GatedConv(64, 128, 3, 1)
        self.conv6 = GatedConv(64, 128, 3, 1)

        # 4x Dilated Gated Convolution
        self.conv7 = GatedConv(64, 128, 3, rate=2)
        self.conv8 = GatedConv(64, 128, 3, rate=4)
        self.conv9 = GatedConv(64, 128, 3, rate=8)
        self.conv10 = GatedConv(64, 128, 3, rate=16)

        # 2x Gated Convolution
        self.conv11 = GatedConv(64, 128, 3, 1)
        self.conv12 = GatedConv(64, 128, 3, 1)

        # Upsample with deconv
        self.conv13 = GatedConv(64, 64, 3, 1, deconv=True)
        self.conv14 = GatedConv(32, 64, 3, 1)
        self.conv15 = GatedConv(32, 32, 3, 1, deconv=True)
        self.conv16 = GatedConv(16, 16, 3, 1)
        self.conv17 = GatedConv(8, 3, 3, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x_stage1 = self.tanh(x)
        return x_stage1


class Critic(nn.Module):
    """
    Critic (Discriminator) Net
    """

    def __init__(self, ch=3):
        super(Critic, self).__init__()
        self.conv_1 = self.conv_block(ch, 64, 5, 2)
        self.conv_2 = self.conv_block(64, 128, 5, 2)
        self.conv_3 = self.conv_block(128, 256, 5, 2)
        self.conv_4 = self.conv_block(256, 256, 5, 2)
        self.conv_5 = self.conv_block(256, 256, 5, 2)
        self.conv_6 = self.conv_block(256, 256, 5, 2)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding=2):
        conv_block = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return conv_block

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = torch.flatten(x)
        return x