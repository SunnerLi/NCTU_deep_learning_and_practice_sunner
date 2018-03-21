from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.optim import SGD
from torch.nn import init
from model import ResNet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import argparse
import torch
import os

# Define parameters
resnet_training_loss_list = []
resnet_test_error_list = []
vellina_training_loss_list = []
vellina_test_error_list = []

def kaiming_init(model):
    """
        Kaiming init can just deal with the tensor whose rank is >= 2.
        For the bias term (rank = 1), use constant initialization
    """
    for m in model.parameters():
        if len(m.size()) > 1:
            nn.init.kaiming_normal(m)
        else:
            nn.init.normal(m, std=0.1)

def train(args):
    global resnet_training_loss_list
    global resnet_test_error_list
    global vellina_training_loss_list
    global vellina_test_error_list

    # Define loader and loss
    training_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download = True, 
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
        ])), 
        batch_size = args.batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, 
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
        ])), 
        batch_size = args.batch_size, shuffle=False, num_workers=2
    )
    criterion = nn.CrossEntropyLoss()   

    # Define ResNet and optimizer
    if (args.block - 2) % 6 != 0:
        raise Exception('You should assign correct number of blockes')
    resnet = ResNet(block_num = (args.block - 2) // 6, skip_connection = True)
    kaiming_init(resnet)
    resnet = resnet.cuda() if torch.cuda.is_available() else resnet
    resnet_optimizer = SGD(resnet.parameters(), lr=0.1, momentum = 0.9, weight_decay = 0.0001)
    resnet_scheduler = MultiStepLR(resnet_optimizer, [81, 122], gamma = 0.1)


    # Define vanilla CNN and optimizer
    vanilla_cnn = ResNet(block_num = (args.block - 2) // 6, skip_connection = False)
    kaiming_init(vanilla_cnn)
    vanilla_cnn = vanilla_cnn.cuda() if torch.cuda.is_available() else vanilla_cnn
    vanilla_optimizer = SGD(vanilla_cnn.parameters(), lr=0.1, momentum = 0.9, weight_decay = 0.0001)
    vanilla_scheduler = MultiStepLR(vanilla_optimizer, [81, 122], gamma = 0.1)

    # Train
    for epoch in range(args.epoch):
        print(' Epoch: %3d ' % (epoch), end = '\t')
        resnet_scheduler.step()
        vanilla_scheduler.step()
        resnet_loss_sum = 0.0
        vanilla_loss_sum = 0.0
        for x, y in training_loader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # -------------------------------------
            # Vanilla CNN part
            # -------------------------------------
            # forward
            vanilla_cnn.train()
            y_ = vanilla_cnn(x)

            # Get training loss and acc
            loss = criterion(y_, y)
            vanilla_loss_sum += loss.data.cpu().numpy()[0]

            # backward
            vanilla_optimizer.zero_grad()
            loss.backward()
            vanilla_optimizer.step()

            # -------------------------------------
            # ResNet part
            # -------------------------------------
            # forward
            resnet.train()
            y_ = resnet(x)

            # Get training loss and acc
            loss = criterion(y_, y)
            resnet_loss_sum += loss.data.cpu().numpy()[0]

            # backward
            resnet_optimizer.zero_grad()
            loss.backward()
            resnet_optimizer.step()

            

        # Test and record
        resnet_test_acc = test(resnet, test_loader, criterion)
        vanilla_test_acc = test(vanilla_cnn, test_loader, criterion)
        print('ResNet Training Loss: %5.3f | Testing Acc: %3.2f' % (resnet_loss_sum, resnet_test_acc), end = '\t')
        print('Vanilla CNN Training Loss: %5.3f | Testing Acc: %3.2f' % (vanilla_loss_sum, vanilla_test_acc))
        resnet_training_loss_list.append(resnet_loss_sum)
        resnet_test_error_list.append(100.0 - resnet_test_acc)
        vellina_training_loss_list.append(vanilla_loss_sum)
        vellina_test_error_list.append(100.0 - vanilla_test_acc)

    # Save
    plt.plot(range(len(resnet_training_loss_list)), resnet_training_loss_list, '-', label='ResNet training loss curve')
    plt.plot(range(len(vellina_training_loss_list)), vellina_training_loss_list, '-', label='Vanilla CNN training loss curve')
    plt.legend()
    plt.savefig('resnet-' + str(args.block) + '_training_loss_curve.png')
    plt.gca().clear()
    plt.plot(range(len(resnet_test_error_list)), resnet_test_error_list, '-', label='ResNet test error curve')
    plt.plot(range(len(vellina_test_error_list)), vellina_test_error_list, '-', label='Vanilla CNN test error curve')
    plt.legend()    
    plt.savefig('resnet-' + str(args.block) + '_test_error_curve.png')
    plt.gca().clear()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    model_path = os.path.join(args.save_dir, 'resnet-' + str(args.block) + '.ckpt')
    torch.save(resnet, model_path)    
    torch.save(vanilla_cnn, model_path)    
    print('ResNet final test error: %5.3f \t Vanilla CNN final test error: %5.3f' % (100.0 - resnet_test_acc, 100.0 - vanilla_test_acc))

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