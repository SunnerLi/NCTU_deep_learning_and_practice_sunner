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
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--label', type=int, default=1, metavar='N',
                        help='The label you want to generate')
    parser.add_argument('--num', type=int, default=1, metavar='N',
                        help='The number of image you want to generate')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.path = './infoGAN_result'
    return args

def test(args):
    if not os.path.exists(args.path):
        raise Exception('You should train first...')
    model = InfoGAN(args.path)
    if os.path.exists(args.path):
        model.load()
    model.cuda()
    model.generateCustom(
        label = args.label, 
        num   = args.num, 
        path  = os.path.join(args.path, 'generate.png')
    )

if __name__ == '__main__':
    args = parse()
    test(args)