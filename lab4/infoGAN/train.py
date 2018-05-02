from torchvision.utils import save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.autograd import Variable
from model import InfoGAN
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import os

def parse():
    parser = argparse.ArgumentParser(description='InfoGAN training')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.path = './infoGAN_result'
    return args

def train(args):
    # Define data loader
    dataloader = DataLoader(
        dset.MNIST(
            './data', 
            transform = transforms.Compose([
                transforms.Resize([64, 64]), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]), 
            download = True
        ), 
        batch_size = args.batch_size, 
        shuffle = True, 
        num_workers = 1
    )

    # Define model
    model = InfoGAN(args.path)
    if os.path.exists(args.path):
        model.load()
    model.cuda()

    # Check if the folder is exist
    if not os.path.exists(args.path):
        os.mkdir(args.path)

    # --------------------------------------------------------------------
    # Train
    # --------------------------------------------------------------------
    generator_loss_list = []
    discriminator_loss_list = []
    posterior_loss_list = []
    real_data_prob_list = []
    fake_data_before_prob_list = []
    fake_data_after_prob_list = []
    for epoch in range(args.epochs):
        for num_iters, (x, _) in enumerate(dataloader, 0):
            model(x)
            model.backward()
            if num_iters % args.log_interval == 0:
                # Show info and record
                info = model.getInfo()
                print("Epoch: %d \t Iter: %d \t loss_G: %.3f \t loss_D: %.3f \
                    loss_Q: %.3f \t prob_real: %.3f \t prob_before: %.3f \t \
                    prob_after: %.3f" % (epoch, num_iters, info['loss_G'], \
                    info['loss_D'], info['loss_Q'], info['prob_real'], \
                    info['prob_before'], info['prob_after']
                ))

                # Append
                generator_loss_list.append(info['loss_G'])
                discriminator_loss_list.append(info['loss_D'])
                posterior_loss_list.append(info['loss_Q'])
                real_data_prob_list.append(info['prob_real'])
                fake_data_before_prob_list.append(info['prob_before'])
                fake_data_after_prob_list.append(info['prob_after'])

                # Generate image
                model.generateGrid(os.path.join(args.path, 'c1.png'))

    # Plot loss curve
    plt.plot(range(len(generator_loss_list)), generator_loss_list, '-', label = 'G loss')
    plt.plot(range(len(discriminator_loss_list)), discriminator_loss_list, '-', label = 'D loss')
    plt.plot(range(len(posterior_loss_list)), posterior_loss_list, '-', label = 'Q loss')
    plt.title('Loss curve')
    plt.legend()
    plt.show()

    # Plot probability curve
    plt.plot(range(len(real_data_prob_list)), real_data_prob_list, '-', label = 'real prob')
    plt.plot(range(len(fake_data_before_prob_list)), fake_data_before_prob_list, '-', label = 'fake prob (before)')
    plt.plot(range(len(fake_data_after_prob_list)), fake_data_after_prob_list, '-', label = 'fake prob (after)')
    plt.title('Probability curve')
    plt.legend()
    plt.show()

    model.save()

if __name__ == '__main__':
    args = parse()
    train(args)