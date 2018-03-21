import torch.nn.functional as F
import torch.nn as nn
import torch

class ResidualBasicBlock(nn.Module):
    def __init__(self, in_channels = 16, out_channels = 16, stride = 1, skip_connection = True):
        super(ResidualBasicBlock, self).__init__()
        self.skip_connection = skip_connection
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        # Deal with channel number inconsistent
        if self.skip_connection:
            self.shortcut = nn.Sequential()
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        if self.skip_connection:
            residual = self.layers(x)
            return F.relu(residual + self.shortcut(x))
        else:
            return F.relu(self.layers(x))