from torchvision import datasets, transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn, optim
from model import CVAE
import torch.utils.data
import numpy as np
import argparse
import torch
import os

def parse():
    parser = argparse.ArgumentParser(description='CVAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.folder = 'cvae_result/'
    return args

def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x.view(-1, 784), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train(epoch, model, loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(loader):
        data = Variable(data).cuda() if torch.cuda.is_available() else Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(loader.dataset)))
    return train_loss / len(loader.dataset)

def main(args):
    # Check if the output folder is exist
    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    # Load data
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # Load model
    model = CVAE().cuda() if torch.cuda.is_available() else CVAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train and generate sample every epoch
    loss_list = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        _loss = train(epoch, model, train_loader, optimizer)
        loss_list.append(_loss)
        model.eval()
        sample = torch.randn(100, 20)
        label = torch.from_numpy(np.asarray(list(range(10)) * 10))
        sample = Variable(sample).cuda() if torch.cuda.is_available() else Variable(sample)
        sample = model.decode(sample, label).cpu()
        save_image(sample.view(100, 1, 28, 28).data, os.path.join(args.folder, 'sample_' + str(epoch) + '.png'), nrow = 10)
    plt.plot(range(len(loss_list)), loss_list, '-o')
    plt.savefig(os.path.join(args.folder, 'cvae_loss_curve.png'))
    torch.save(model.state_dict(), os.path.join(args.folder, 'cvae.pth'))

if __name__ == '__main__':
    args = parse()
    main(args)