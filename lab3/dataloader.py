import torch.utils.data as data
import multiprocessing
import numpy as np
import random
import h5py
import json
import os

class DataLoader(data.Dataset):
    def reset