from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.optim import Adam
from dataloader import *
from dataloaderraw import *
import misc.utils as utils
import torch.nn as nn
import numpy as np
import eval_utils
import pickle
import models
import torch
import time
import opts
import cv2

"""
    This script will visualize one image in the specific folder
    
    run: python3 visualize.py --image_folder visualize_img --caption_model show_attend_tell
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-0.1*x))

def maskMul(mask, arr):
    arr = (arr / 255).astype(np.float32)
    # print(np.shape(arr), np.min(arr), np.max(arr))
    result = np.copy(arr)
    h, w, c = np.shape(result)
    # mask = sigmoid(mask)
    for i in range(c):
        result[:, :, i] *= mask
    return result

if __name__ == '__main__':
    opt = opts.parse()
    opt.use_att = utils.if_use_attention(opt.caption_model)
    # loader = DataLoaderRaw(opt)  
    loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': '',
                            'batch_size': opt.batch_size,
                            'cnn_model': 'resnet101'})

    # Load infos
    with open(opt.caption_model + '_infos_or.pkl', 'rb') as f:
        infos = pickle.load(f)
    loader.ix_to_word = infos['vocab']
    loader.vocab_size = len(infos['vocab'])
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    # Set hyper-parameters
    verbose = True
    num_images = 5000
    minimun_loss = 10.00
    split = 'val'
    dataset = 'coco'
    beam_size = 1

    # Load model
    model = models.setup(opt)
    model.load_state_dict(torch.load(opt.caption_model + '_model_or.pth'))
    model.cuda()
    loader.reset_iterator(split)

    # Visualize
    n = 0
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp

        # forward the model to also get generated samples for each image
        seq, _, attentions = model.sample(fc_feats, att_feats, record_attention = True, opt = vars(opt))
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        # attentions = attentions[:, 0, :]
        original_img = data['img_batch'][0]
        attentions = attentions.contiguous().view(attentions.size(0), 14, 14)
        images = attentions.data.cpu().numpy()
        for i in range(15):
            print(sents[0].split()[i])
            image = np.expand_dims(images[i, :, :], -1)
            image = (image * 255)
            image = cv2.resize(image, (196, 196))
            # print(np.shape(original_img), np.min(original_img), np.max(original_img))
            weighted_img = maskMul(cv2.resize(image, (640, 426)), original_img)
            # weighted_img = cv2.resize(image, (640, 426))
            weighted_img = cv2.cvtColor(weighted_img, cv2.COLOR_BGR2RGB)
            cv2.imshow('origin image', original_img)
            cv2.imshow('image with attention', weighted_img)
            cv2.imwrite(str(i) + '.png', weighted_img * 255)
            cv2.waitKey()
        break