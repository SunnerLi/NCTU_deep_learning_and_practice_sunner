import torch.utils.model_zoo as zoo
import torch.nn as nn
import math

__all__ = ['resnet101']

model_urls = {
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth'
}

def conv3x3(in_channels, out_channels, strides = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 3,
        stride = strides, padding = 1, bias = False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides = 1, down_sampling = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = strides, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace = True)
        self.strides = strides
        self.down_sampling = down_sampling

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.down_sampling is not None:
            residual = self.down_sampling(x)
        out += residual
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layer_list, num_class = 1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, padding = 3, stride = 2, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)
        self.layer1 = self._make_layer(block, 64, layer_list[0])
        self.layer2 = self._make_layer(block, 128, layer_list[1], strides = 2)
        self.layer3 = self._make_layer(block, 256, layer_list[2], strides = 2)
        self.layer4 = self._make_layer(block, 512, layer_list[3], strides = 2)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_class)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, num_block, strides = 1):
        down_sampling = None
        if strides != 1 or self.in_channels != channels * block.expansion:
            down_sampling = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size = 1, stride = strides, bias = True),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers = [block(self.in_channels, channels, strides, down_sampling)]
        self.in_channels = channels * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet101(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(zoo.load_url(model_urls['resnet101']))
    return model