from torchvision.utils import save_image
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn, optim
import numpy as np
import itertools
import torch
import os

def np2var(arr, to_float = True):
    if to_float:
        return Variable(torch.from_numpy(arr).float()).cuda()
    else:
        return Variable(torch.from_numpy(arr)).cuda()        

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(74, 512, 4, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.main(x)
        return output

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias = True),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, bias = True),
            nn.Sigmoid()
        )
        self.Q = nn.Sequential(
            nn.Linear(8192, 100, bias = True),
            nn.ReLU(),
            nn.Linear(100, 10, bias = True),
        )

    def forward(self, x):
        fe = self.main(x)
        output_d = self.discriminator(fe).view(-1, 1)
        output_q = self.Q(fe.view(output_d.size(0), -1))
        return output_d, output_q

class InfoGAN(nn.Module):
    def __init__(self, path, batch_size = 32):
        super(InfoGAN, self).__init__()
        self.G = _netG()
        self.D = _netD()
        self.path = path
        self.batch_size = batch_size
        
        # Define optimizer and criterion
        self.optimD = optim.Adam(
            itertools.chain(
                self.D.main.parameters(),self.D.discriminator.parameters()),
            lr=0.0002, betas=(0.5, 0.99)
        )
        self.optimG = optim.Adam(
            itertools.chain(
                self.G.parameters(), self.D.Q.parameters()), 
            lr=0.001, betas=(0.5, 0.99)
        )
        self.criterionD = nn.BCELoss()
        self.criterionQ = nn.CrossEntropyLoss()

    def forward(self, real_x):
        self.real_x = Variable(real_x).cuda()

    def _noise_sample(self, bs):
        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs),idx] = 1.0

        dis_c = Variable(torch.Tensor(c)).cuda()
        noise = np2var(np.random.normal(size = [bs, 64]))
        z = torch.cat([noise, dis_c], 1).view(-1, 74, 1, 1)
        self.noise = z
        self.class_target = np2var(idx, False)

    def backward_D(self):
        # true
        self.prob_real, _ = self.D(self.real_x)
        loss_real = self.criterionD(self.prob_real, torch.ones_like(self.prob_real))

        # fake
        self._noise_sample(self.batch_size)
        fake_x = self.G(self.noise)
        prob_fake, _ = self.D(fake_x)
        loss_fake = self.criterionD(prob_fake, torch.zeros_like(prob_fake))

        # merge
        self.loss_D = loss_real + loss_fake
        self.loss_D.backward()

    def backward_G_and_Q(self, optimizer = None):
        self._noise_sample(self.batch_size)
        fake_x = self.G(self.noise)
        self.prob_fake, self.posterior_fake = self.D(fake_x)
        dis_loss = self.criterionQ(self.posterior_fake, self.class_target)
        reconst_loss = self.criterionD(self.prob_fake, torch.ones_like(self.prob_fake))
        self.loss_G = reconst_loss + dis_loss
        self.loss_Q = dis_loss
        self.loss_G.backward()

        if optimizer is not None:
            optimizer.step()
            fake_x = self.G(self.noise)
            self.prob_after, _ = self.D(fake_x)

    def backward(self):
        self.optimD.zero_grad()
        self.backward_D()
        self.optimD.step()

        self.optimG.zero_grad()
        self.backward_G_and_Q(self.optimG)

    def getInfo(self):
        out = {
            'loss_G': self.loss_G.data[0],
            'loss_D': self.loss_D.data[0],
            'loss_Q': self.loss_Q.data[0],
            'prob_real': self.prob_real.data[0],
            'prob_before': self.prob_fake.data[0],
            'prob_after': self.prob_after.data[0]
        }
        return out

    def generateGrid(self, path):
        idx = np.arange(10).repeat(10)
        idx = np.reshape(np.reshape(idx, [10, 10]).T, -1)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        one_hot = np2var(one_hot)
        fix_noise = Variable(torch.Tensor(100, 64).uniform_(-1, 1)).cuda()

        z = torch.cat([fix_noise, one_hot], 1).view(-1, 74, 1, 1)
        fake_x = self.G(z)
        save_image(fake_x.data, path, nrow=10)

    def generateCustom(self, label, num, path):
        idx = label
        one_hot = np.zeros((num, 10))
        one_hot[range(num), idx] = 1
        one_hot = np2var(one_hot)
        fix_noise = Variable(torch.Tensor(num, 64).uniform_(-1, 1)).cuda()

        z = torch.cat([fix_noise, one_hot], 1).view(-1, 74, 1, 1)
        fake_x = self.G(z)
        save_image(fake_x.data, path, nrow=10)

    def save(self):
        torch.save(self.G.state_dict(), os.path.join(self.path, 'g.pth'))
        torch.save(self.D.state_dict(), os.path.join(self.path, 'd.pth'))

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.path, 'g.pth')))
        self.D.load_state_dict(torch.load(os.path.join(self.path, 'd.pth')))

if __name__ == '__main__':
    model = InfoGAN('./')
    print(model)