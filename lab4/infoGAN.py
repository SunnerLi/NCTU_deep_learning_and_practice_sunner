from torchvision.utils import save_image
from torch.utils.data import DataLoader
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

class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
                (x-mu).pow(2).div(var.mul(2.0)+1e-6)
        return logli.sum(1).mean().mul(-1)

class Trainer:
    def __init__(self, G, FE, D, Q, args):
        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q
        self.args = args

    def _noise_sample(self, dis_c, con_c, noise, bs):
        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs),idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

        return z, idx

    def train(self):
        # Check if the folder is exist
        if not os.path.exists(args.path):
            os.mkdir(args.path)

        # Allocate the input memory (tensor type)
        real_x = torch.FloatTensor(self.args.batch_size, 1, 28, 28).cuda()
        label = torch.FloatTensor(self.args.batch_size).cuda()
        dis_c = torch.FloatTensor(self.args.batch_size, 10).cuda()
        con_c = torch.FloatTensor(self.args.batch_size, 2).cuda()
        noise = torch.FloatTensor(self.args.batch_size, 62).cuda()

        # Transfer the tensor into variable
        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        noise = Variable(noise)

        # Define loss
        criterionD = nn.BCELoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
        criterionQ_con = log_gaussian()

        # Define optimizer
        optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

        # Define data loader
        dataloader = DataLoader(
            dset.MNIST(
                './dataset', 
                transform = transforms.ToTensor(), 
                download = True
            ), 
            batch_size = self.args.batch_size, 
            shuffle = True, 
            num_workers = 1
        )

        # fixed random variables
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])

        idx = np.arange(10).repeat(10)
        idx = np.reshape(np.reshape(idx, [10, 10]).T, -1)
        print(np.shape(idx))
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)

        # --------------------------------------------------------------------
        # Train
        # --------------------------------------------------------------------
        for epoch in range(self.args.epochs):
            for num_iters, batch_data in enumerate(dataloader, 0):
                # real part
                optimD.zero_grad()

                x, _ = batch_data

                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs)
                dis_c.data.resize_(bs, 10)
                con_c.data.resize_(bs, 2)
                noise.data.resize_(bs, 62)

                real_x.data.copy_(x)
                fe_out1 = self.FE(real_x)
                probs_real = self.D(fe_out1)
                label.data.fill_(1)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z, idx = self._noise_sample(dis_c, con_c, noise, bs)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = self.Q(fe_out)
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                dis_loss = criterionQ_dis(q_logits, target)
                con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1

                G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()

                if num_iters % self.args.log_interval == 0:
                    
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                      epoch, num_iters, D_loss.data.cpu().numpy(),
                      G_loss.data.cpu().numpy())
                    )

                    noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))

                    con_c.data.copy_(torch.from_numpy(c1))
                    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
                    x_save = self.G(z)
                    print(x_save.size())
                    save_image(x_save.data, os.path.join(args.path, 'c1.png'), nrow=10)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def parse():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
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
        i.apply(weights_init)

    trainer = Trainer(g, fe, d, q, args)
    trainer.train()