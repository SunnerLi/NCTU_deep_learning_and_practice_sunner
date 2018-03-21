from module import ResidualBasicBlock
import torch.nn.functional as F
import torch.nn as nn
import torch

class ResNet(nn.Module):
    def __init__(self, block_num = 3, skip_connection = True):
        # Initalize
        super(ResNet, self).__init__()
        self.block_num = block_num
        self.layers = []

        # 1st conv
        self.layers += [nn.Conv2d(3, 16, 7, padding=3)]

        # Residual block
        prev_channel = 16
        curr_channel = 16
        for i in range(3):
            for j in range(block_num):
                curr_channel = curr_channel * 2 if curr_channel != (2 ** (i + 4)) else curr_channel
                stride = 2 if (j == 0 and i > 0) else 1
                self.layers += [ResidualBasicBlock(prev_channel, curr_channel, stride, skip_connection)]
                prev_channel = prev_channel * 2 if i != 0 and j == 0 else prev_channel
        
        # Last
        self.layers = nn.Sequential(*self.layers)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layers(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size()[0], -1)
        return self.fc(x)