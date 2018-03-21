import torch.nn.functional as F
import torch.nn as nn
import torch

class ResidualBasicBlock(nn.Module):
    def __init__(self, in_channels = 16, out_channels = 16, stride = 1):
        super(ResidualBasicBlock, self).__init__()
        self.layers = []
        self.layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
        ]
        self.layers = nn.Sequential(*self.layers)

        # Deal with channel number inconsistent
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.layers(x)
        return F.relu(residual + self.shortcut(x))