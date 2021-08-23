import torch
from torch import nn
from torch.nn.utils import spectral_norm


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size, padding=1, padding_mode='replicate')
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.doubleconv = DoubleConv(in_ch, out_ch)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.doubleconv(x)
        x = self.maxpool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, strat='trans_conv'):
        super().__init__()
        if strat == 'trans_conv':
            self.upsample = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.doubleconv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.upsample(x)
        x = self.doubleconv(x)
        return x


class UNet(nn.Module):
    """"
    A UNet without skip connections, uses transposed convolutions as default upsampling strategy.
    Currently using padding to prevent the feature maps from getting to small to upsample again.
    input: multiple vertebrae in diff channels shape = (context*2, patch_size, patch_size, patch_size)
    output: shape to be predicted shape = (1, patch_size, patch_size, patch_size)
    """
    def __init__(self, in_channels=4, out_channels=1, init_filters=64):
        super().__init__()

        # encoder
        self.down1 = DownBlock(in_channels, init_filters)
        self.down2 = DownBlock(init_filters, init_filters*2)
        self.down3 = DownBlock(init_filters*2, init_filters*4)
        self.down4 = DownBlock(init_filters*4, init_filters*8)

        # decoder
        self.up1 = UpBlock(init_filters*8, init_filters*4)
        self.up2 = UpBlock(init_filters*4, init_filters*2)
        self.up3 = UpBlock(init_filters*2, init_filters)
        self.up4 = UpBlock(init_filters, out_channels)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator Net (global)
    Input: patch (5, patch_size, patch_size, patch_size)
    Output: validity of the sample => whether it is from real data or generated by the UNet
    """

    def __init__(self, ch=5):
        super(Discriminator, self).__init__()
        self.conv_1 = self.conv_block(ch, 64, 5, 2)
        self.conv_2 = self.conv_block(64, 128, 5, 2)
        self.conv_3 = self.conv_block(128, 256, 5, 2)
        self.conv_4 = self.conv_block(256, 256, 5, 2)
        self.conv_5 = self.conv_block(256, 256, 5, 2)
        self.conv_6 = self.conv_block(256, 256, 5, 2)
        self.fc = nn.Linear(256, 1)

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x