from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

class MyResnet(nn.Module):
    def __init__(self, resnet):
        super(MyResnet, self).__init__()
        self.resnet = resnet

    def forward(self, x, att_size = 14):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, dim = 0)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        fc = x.mean(-1).mean(-1).squeeze()
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)
        return fc, att
