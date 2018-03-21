from module import ResidualBasicBlock
import torch.nn as nn
import torch

class ResNet(nn.Module):
    def __init__(self, block_num = 3):
        # Initalize
        super(ResNet, self).__init__()
        self.block_num = block_num
        self.layers = []

        # 1st conv
        self.layers += [nn.Conv2d(3, 16, 3, padding=1)]

        # Residual block
        prev_channel = 16
        curr_channel = 16
        for i in range(3):
            for j in range(block_num):
                curr_channel = curr_channel * 2 if curr_channel != (2 ** (i + 4)) else curr_channel
                self.layers += [ResidualBasicBlock(prev_channel, curr_channel, 1)]
                prev_channel = prev_channel * 2 if i != 0 and j == 0 else prev_channel
            self.layers += [nn.MaxPool2d(kernel_size = 2)]
        
        # Last
        self.layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.layers = nn.Sequential(*self.layers)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 64)
        return self.fc(x)