from torchvision.utils import save_image
from torch.autograd import Variable
from model import CVAE
import numpy as np
import argparse
import torch
import os

def parse():
    parser = argparse.ArgumentParser(description='CVAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--digits', type=int, default=1, metavar='N',
                        help='The digit you want to generate')
    parser.add_argument('--num', type=int, default=1, metavar='N',
                        help='The number of image you want to generate')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.folder = 'cvae_result/'
    return args

def main(args):
    # Check if the output folder is exist
    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    # Load model
    model = CVAE().cuda() if torch.cuda.is_available() else CVAE()
    model.load_state_dict(torch.load(os.path.join(args.folder, 'cvae.pth')))

    # Generate
    sample = torch.randn(args.num, 20)
    label = torch.from_numpy(np.asarray([args.digits] * args.num))
    sample = Variable(sample).cuda() if torch.cuda.is_available() else Variable(sample)
    sample = model.decode(sample, label).cpu()
    save_image(sample.view(args.num, 1, 28, 28).data, os.path.join(args.folder, 'generate.png'), nrow = 10)

if __name__ == '__main__':
    args = parse()
    main(args)