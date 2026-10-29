"""
Module unet

Contains U-Net based model for singal decomposition

Authors:

Date created:
    18/06/2025

Adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet

NOTE:
    "channels" terminology can be confusing in this context.
    We are using "channels" to refer to the number of features at each time step,
    while "height" refers to the number of emg channels and "width" refers to the number of time steps.
"""

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.visualisation as vis

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2

    This version uses circular padding in the height (emg channels) dimension.
    
    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, out_channels, height, width)
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        conv_pad = (0, 1)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=conv_pad, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=conv_pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.circular_pad_h = (0, 0, 1, 1)
        

    def forward(self, x):
        x = F.pad(x, self.circular_pad_h, mode='circular')
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.pad(out, self.circular_pad_h, mode='circular')
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out



class Down(nn.Module):
    """Downscaling with maxpool then double conv
    
    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, out_channels, height // 2, width // 2)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv
    
    Input shape:
        x1: (batch_size, in_channels, in_height, in_width)
        x2: (batch_size, skip_channels, out_height, out_width) 

    Generally:
        out_height = in_height * 2
        out_width = in_width * 2
        skip_channels = in_channels // 2

    Output shape: (batch_size, out_channels, out_height, out_width) 
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UnetEncoder(nn.Module):
    """
    Module to endcode a sequence of emg/muap windows to a vector using a 1D conv network

    Input shape: (B, in_channels, m, T)
    Output shape: (B, out_channels, m//8, T//8)
    """
    def __init__(self, in_channels=2, out_channels=512):
        super(UnetEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, out_channels))

    def forward(self, x, return_hidden=True):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if return_hidden:  
            return x1, x2, x3, x4
        
        return x4


class UnetDecoder(nn.Module):
    """
    Module to decode an of emg/muap vecotr representations to muap using a 1D conv network

    Input shape: (B, in_channels, m//8, T//8) OR (B*W, in_channels, m//8, T//8)
    Output shape: (B, out_channels, m, T)
    """
    def __init__(self, in_channels=512, out_channels=1):
        super(UnetDecoder, self).__init__()

        self.up1 = (Up(in_channels, 256))
        self.up2 = (Up(256, 128))
        self.up3 = (Up(128, 64))
        self.outc = (OutConv(64, out_channels))


    def forward(self, x1, x2, x3, x4):

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class UNet2d(nn.Module):
    """
    Input shape: (batch_size, in_channels, m, T)
    Output shape: (batch_size, out_channels, m, T)
    """

    def __init__(self, cfg):
        super(UNet2d, self).__init__()

        in_channels = 2
        out_channels = 1
        
        self.encoder = UnetEncoder(in_channels, out_channels=512)
        self.decoder = UnetDecoder(in_channels=512, out_channels=out_channels)

    def forward(self, x):
        """
        Args: 
            x: (batch_size, n_channels, time_steps)
        """
        x1, x2, x3, x4 = self.encoder(x, return_hidden=True)
        logits = self.decoder(x1, x2, x3, x4)
        logits = logits.squeeze(1)  # Reshape back to (B, n_channels, time_steps)
        return logits  # (B, n_channels, time_steps)


if __name__ == "__main__":
    # Input shape: (batch_size, in_channels, time_steps)
    x = torch.randn(1, 16, 34)  # batch of 1, 16 channels, 33 time steps
    x = x.unsqueeze(1)  # Reshape to (B, 1, m, T)
    model = Down(1, 32)
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    x = torch.randn(1, 5, 16, 33)  # batch of 1, 5 "channels", 16 channels, 33 time steps
    model = DoubleConv(5, 32)
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    x = torch.randn(1, 2, 16, 34)  # batch of 1, 16 channels, 33 time steps
    model =  nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(2, 32)
        )
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    x1 = torch.randn(1, 16, 2, 33)  # batch of 1, 16 channels, 33 time steps
    x2 = torch.randn(1, 8, 5, 60)  # batch of 1, 32 channels, 33 time steps
    model = Up(16, 32, bilinear=False)
    y = model(x1, x2)
    print(f"Input shape:  {x1.shape}")
    print(f"Skip shape:  {x2.shape}")
    print(f"Output shape: {y.shape}")


    # print("Testing UNet")
    # x = torch.randn(1, 12, 64)  # batch of 1, 16 channels, 33 time steps
    # model = UNet(1, 1)
    # y = model(x)
    # print(f"Input shape:  {x.shape}")
    # print(f"Output shape: {y.shape}")