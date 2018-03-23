from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.optim import SGD, Adam
from main import kaiming_init
from torch.nn import init
from model import Pre_actResNet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import argparse
import torch
import os

# Define parameters
preact_resnet_training_loss_list = []
preact_resnet_test_error_list = []

def train(args):
    global preact_resnet_training_loss_list
    global preact_resnet_test_error_list

    # Define loader and loss
    training_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download = True, 
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
        ])), 
        batch_size = args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, 
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
        ])), 
        batch_size = args.batch_size, shuffle=True, num_workers=2
    )
    criterion = nn.CrossEntropyLoss()  

    # Define Pre-act ResNet and optimizer
    if (args.block - 2) % 6 != 0:
        raise Exception('You should assign correct number of blockes')
    net = Pre_actResNet(block_num = (args.block - 2) // 6)
    kaiming_init(net)
    net = net.cuda() if torch.cuda.is_available() else net
    optimizer = SGD(net.parameters(), lr=0.1, momentum = 0.9, weight_decay = 0.0001)
    scheduler = MultiStepLR(optimizer, [80, 120], gamma = 0.2) 

    # Train
    for epoch in range(args.epoch):
        print(' Epoch: %3d ' % (epoch), end = '\t')
        scheduler.step()
        loss_sum = 0.0
        for x, y in training_loader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # forward
            net.train()
            y_ = net(x)

            # Get training loss and acc
            loss = criterion(y_, y)
            loss_sum += loss.data.cpu().numpy()[0]

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test and record
        test_acc = test(net, test_loader, criterion)
        print('ResNet Training Loss: %5.3f | Testing Acc: %3.2f' % (loss_sum, test_acc))
        preact_resnet_training_loss_list.append(loss_sum)
        preact_resnet_test_error_list.append(100.0 - test_acc)

    # Save
    plt.plot(range(len(preact_resnet_training_loss_list)), preact_resnet_training_loss_list, '-', label='Pre-act ResNet training loss curve')
    plt.legend()
    plt.savefig('Pre-act_resnet-' + str(args.block) + '_training_loss_curve.png')
    plt.gca().clear()
    plt.plot(range(len(preact_resnet_test_error_list)), preact_resnet_test_error_list, '-', label='Pre-act ResNet test error curve')
    plt.legend()    
    plt.savefig('Pre-act_resnet-' + str(args.block) + '_test_error_curve.png')
    plt.gca().clear()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    model_path = os.path.join(args.save_dir, 'pre-act_resnet-' + str(args.block) + '.ckpt')
    torch.save(net, model_path)    
    print('Pre-act ResNet final test error: %5.3f' % (100.0 - test_acc))

def test(net, loader, criterion):
    loss_sum = 0.0
    correct_sum = 0.0
    total = 0
    for x, y in loader:
        # forward
        total += x.cpu().size()[0]        
        net.eval()
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        y_ = net(x)

        # Get loss and acc
        loss_sum += criterion(y_, y).data.cpu().numpy()[0]
        _, pred = torch.max(y_.data, 1)        
        correct_sum += pred.eq(y.data).cpu().sum()
    return correct_sum * 100 / total 

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', default = 20, type = int, help = 'number of residual block')
    parser.add_argument('--batch_size', default = 32, type = int, help = 'batch size')
    parser.add_argument('--epoch', default = 1, type = int, help = 'the number of training epoch')
    parser.add_argument('--save_dir', default = './', type = str, help = 'the folder of your model')
    args = parser.parse_args()
    train(args)