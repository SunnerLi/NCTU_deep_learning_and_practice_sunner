from .CaptionModel import CaptionModel
from torch.autograd import *
import misc.utils as utils
import torch.nn.functional as F
import torch.nn as nn
import torch

class ShowTellModel(CaptionModel):
    def __init__(self, opt):
        super(ShowTellModel, self).__init__()

        # Define parameters
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = 512
        self.rnn_type = 'GRU'
        self.rnn_size = 512
        self.num_layers = opt.num_layers
        self.dropout_prob = 0.5
        self.seq_length = opt.seq_length
        self.fc_feat_size = 2048
        self.schedule_sampling_prob = 0.0

        # Define layers
        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = nn.GRU(self.input_encoding_size, self.rnn_size, self.num_layers, bias = False, dropout = self.dropout_prob)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.num_layers, batch_size, self.rnn_size).zero_())

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        output_list = []
        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.schedule_sampling_prob > 0.0:
                    raise Exception("This case is not considered now...")
                else:
                    it = seq[:, i-1].clone()
                if i >= 2 and seq[:, i-1].data.sum() == 0:
                    break
                xt = self.embed(it)

            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))
            output_list.append(output)
        return torch.cat([_.unsqueeze(1) for _ in output_list[1:]], 1).contiguous()

    def get_logprob_state(self, it, state):
        xt = self.embed(it)
        output, state = self.core(xt.unsqueeze(0), state)
        logprob = F.log_softmax(self.logit(self.dropout(output.squeeze(0))))
        return logprob, state