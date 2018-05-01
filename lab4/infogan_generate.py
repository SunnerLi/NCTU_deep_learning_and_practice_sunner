from torchvision.utils import save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.autograd import Variable
from model import FrontEnd, G, D, Q
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import os

class Tester:
    def __init__(self, G, FE, D, Q, args):
        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q
        self.args = args

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(args.path, 'g.pth')))
        self.FE.load_state_dict(torch.load(os.path.join(args.path, 'fe.pth')))
        self.D.load_state_dict(torch.load(os.path.join(args.path, 'd.pth')))
        self.Q.load_state_dict(torch.load(os.path.join(args.path, 'q.pth')))

    def _noise_sample(self, dis_c, con_c, noise, bs):
        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs),idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

        return z, idx

    def generate(self):
        # Allocate the input memory (tensor type)
        dis_c = torch.FloatTensor(self.args.num, 10).cuda()
        con_c = torch.FloatTensor(self.args.num, 2).cuda()
        noise = torch.FloatTensor(self.args.num, 62).cuda()

        # Transfer the tensor into variable
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        noise = Variable(noise)

        # Generate one hot vector
        one_hot = np.zeros((args.num, 10))
        one_hot[range(args.num), args.label] = 1

        # Fixed random variables
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])

        # Work
        noise.data.copy_(torch.Tensor(args.num, 62).uniform_(-1, 1))
        dis_c.data.copy_(torch.Tensor(one_hot))
        print(np.shape(c1))
        con_c = Variable(torch.rand(con_c.size())).cuda()
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
        x_save = self.G(z)
        save_image(x_save.data, os.path.join(args.path, 'generate.png'), nrow=10)

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

if __name__ == '__main__':
    args = parse()
    fe = FrontEnd()
    d = D()
    q = Q()
    g = G()

    for i in [fe, d, q, g]:
        i.cuda()

    tester = Tester(g, fe, d, q, args)
    tester.load()
    tester.generate()