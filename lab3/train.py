from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.optim import Adam
from dataloader import *
import misc.utils as utils
import torch.nn as nn
import eval_utils
import pickle
import models
import torch
import time
import opts

if __name__ == '__main__':
    opt = opts.parse()
    opt.use_att = utils.if_use_attention(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    # Train from scratch
    iteration = 0
    epoch = 0
    loss_list = []
    update_lr_flag = False

    # Load model
    model = models.setup(opt)
    model.cuda()
    criterion = utils.LanguageModelCriterion()
    optimizer = Adam(model.parameters(), lr = 0.0004)

    while True:
        if update_lr_flag:
            # don't implement lr decay now...
            pass
        else:
            opt_current_lr = 0.0004

        # Load data and record time
        start = time.time()
        data = loader.get_batch('train')
        load_data_time = time.time() - start
        torch.cuda.synchronize()

        # Transfer batch data as variable object
        fc_feats = Variable(torch.from_numpy(data['fc_feats']), requires_grad = False).cuda()
        att_feats = Variable(torch.from_numpy(data['att_feats']), requires_grad = False).cuda()
        labels = Variable(torch.from_numpy(data['labels']), requires_grad = False).cuda()
        masks = Variable(torch.from_numpy(data['masks']), requires_grad = False).cuda()

        # forward & backward
        optimizer.zero_grad()
        loss = criterion(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:])
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss = loss.data[0]
        loss_list.append(train_loss)
        torch.cuda.synchronize()
        forward_backward_time = time.time() - start
        print("Epoch : {} \t Iter : {} \t Train_loss : {:.3f} \t Load data time : {:.3f} \t Train time : {:.3f} \r" \
            .format(epoch, iteration, train_loss, load_data_time, forward_backward_time))

        # Update iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True            

        if epoch > opt.epoch:
        # if iteration > 10:
            break

    # Validate
    eval_arg = {'split': 'val', 'dataset': 'data/cocotalk.json'}
    eval_arg.update(vars(opt))
    val_loss, predictions, lang_states, minimun_loss = eval_utils.eval_split(model, criterion, loader, eval_arg)
    print('Final minimun validation loss: {:.3f}'.format(minimun_loss))
    plt.plot(range(len(loss_list)), loss_list, label = 'Training loss curve')
    plt.legend()
    plt.show()