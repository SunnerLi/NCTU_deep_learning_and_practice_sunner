from torch.autograd import Variable
from torch.optim import Adam
from dataloader import *
import misc.utils as utils
import torch.nn as nn
import pickle
import models
import torch
import opts

if __name__ == '__main__':
    opt = opts.parse()
    opt.use_att = utils.if_use_attention(opt.caption_model)
    loader = DataLoader(opt)