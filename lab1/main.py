from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.optim import SGD
from torch.nn import init
from model import ResNet
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import argparse
import torch

def kaiming_init(model):
    """
        Kaiming init can just deal with the tensor whose rank is >= 2.
        For the bias term (rank = 1), use constant initialization
    """
    for m in model.parameters():
        if len(m.size()) > 1:
            nn.init.kaiming_normal(m)
        else:
            nn.init.constant(m, 0.1)

def train(args):
    # Define parameters
    loss_list = []

    # Define loader
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download = False, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle=False, num_workers=2)

    # Define model, loss and optimizer
    if (args.block - 2) % 6 != 0:
        raise Exception('You should assign correct number of blockes')
    net = ResNet(block_num = (args.block - 2) // 6)
    kaiming_init(net)
    net = net.cuda() if torch.cuda.is_available() else net
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.1, momentum = 0.9, weight_decay = 0.0001)
    scheduler = MultiStepLR(optimizer, [81, 122], gamma = 0.1)

    # Train
    for epoch in range(args.epoch):
        print(' Epoch: %3d ' % (epoch), end = '\t')
        scheduler.step()
        loss_sum = 0.0
        correct_sum = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            # forward
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            y_ = net(x)

            # Get loss and acc
            loss = criterion(y_, y)
            loss_sum += loss.data.cpu().numpy()[0]
            _, pred = torch.max(y_.data, 1)
            correct = pred.eq(y.data).cpu().sum()
            correct_sum += correct

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Loss: %.3f | Acc: %3.2f ' % (loss_sum, correct_sum / 600))


def test(args):
    pass

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', default = 20, type = int, help = 'number of residual block')
    parser.add_argument('--batch_size', default = 32, type = int, help = 'batch size')
    parser.add_argument('--epoch', default = 1, type = int, help = 'the number of training epoch')
    parser.add_argument('--train', default = True, type = bool, help = 'True is train step, False is test step')
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        test(args)