import torch.utils.data as data
import multiprocessing
import numpy as np
import random
import h5py
import json
import os

class DataLoader(data.Dataset):
    def print(self, *arg):
        print('< DataLoader > - ', *arg)

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix2word

    def get_seq_length(self):
        return self.seq_length

    def read_files(self):
        self.feats_fc = h5py.File(os.path.join(self.opt.input_fc_dir, 'feats_fc.h5'), 'r')
        self.feats_att = h5py.File(os.path.join(self.opt.input_att_dir, 'feats_att.h5'), 'r')

    def get_data(self, ix):
        self.read_files()
        pass#????

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = 5
        self.use_att = getattr(self.opt, 'use_att', True)

        # Load JSON file
        self.print('Loading JSON file: ', 'data/cocotalk.json')
        self.info = json.load(open('data/cocotalk.json'))
        self.ix2word = self.info['ix_to_word']
        self.vocab_size = len(self.ix2word)
        self.print('Vocabulary size  : ', self.vocab_size)

        # Open hdf5 file
        self.print('Loading hdf5 file: ', self.opt.input_fc_dir, \
            self.opt.input_att_dir, self.opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver = 'core')
        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir

        # Load sequence data
        # * label = (N, T, F)
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        self.print('Max seq length   : ', self.seq_length)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        self.num_img = self.label_start_ix.shape[0]
        self.print('Number of image  : ', self.num_img)

        # Split the dataset as train, val and test
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif self.opt.train_only == 0:
                self.split_ix['train'].append(ix)
        self.print('Num of train img : ', len(self.split_ix['train']))
        self.print('Num of val   img : ', len(self.split_ix['val']))
        self.print('Num of test  img : ', len(self.split_ix['test']))

        # Define iterator and 3 blobFetcher to load image
        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        self._prefetch_process = {}
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')

        # Define terminate task
        def cleanUp():
            for split in self._prefetch_process.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanUp)

    def get_batch(self, split, batch_size = None, seq_per_img = None):
        # Initialization
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img
        fc_batch = []
        att_batch = []
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')
        wrapped = False
        infos = []
        gts = []

        # -------------------------------------------------------------------------
        # Filled batch data
        # -------------------------------------------------------------------------
        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch += [tmp_fc] * seq_per_img
            att_batch += [tmp_att] * seq_per_img

            # fetch label
            ix1 = self.label_start_ix[ix] - 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1
            if ncap <= 0: raise Exception('ncap error!') 

            # check if the caption sentence is enough
            if ncap < seq_per_img:
                seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
                for q in range(seq_per_img):
                    ix1 = random.randint(ix1, ix2)
                    seq[q, :] = self.h5_label_file['labels'][ix1, :self.seq_length]
            else:
                ix1 = random.randint(ix1, ix2 - seq_per_img + 1)
                seq = self.h5_label_file['labels'][ix1: ix1 + seq_per_img, :self.seq_length]
            label_batch[i * seq_per_img: (i+1) * seq_per_img, 1: self.seq_length + 1] = seq
            if tmp_wrapped:
                wrapped = True

            # reward
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # record associated info
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # Generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, label_batch)))
        for ix, row in enumerate(label_batch):
            row[:nonzeros[ix]] = 1

        # Record final batch info
        data = {}
        data['fc_feats'] = np.stack(fc_batch)
        data['att_feats'] = np.stack(att_batch)
        data['labels'] = label_batch
        data['gts'] = gts
        data['masks'] = mask_batch
        data['bound'] = {
            'it_pos_now': self.iterators[split],
            'it_max': len(self.split_ix[split]),
            'wrapped': wrapped
        }
        data['infos'] = infos
        return data

    def __getitem__(self, index):
        return self.get_data(index)

    def __len__(self):
        return len(self.info['images'])

class BlobFetcher():
    def __init__(self, split, data_loader, if_shuffle = False):
        self.split = split
        self.data_loader = data_loader
        self.if_shuffle = if_shuffle

    def reset(self):
        sampler = self.data_loader.split_ix[self.split][self.data_loader.iterators[self.split]:]
        self.split_loader = iter(data.DataLoader(
            dataset = self.data_loader,
            batch_size = 1,
            sampler = sampler,
            shuffle = False,
            pin_memory = True,
            num_workers = multiprocessing.cpu_count(),
            collate_fn = lambda x: x[0]
        ))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()
        return tmp + [wrapped]