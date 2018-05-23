from torch.autograd import Variable
import numpy as np
import torch

def to_var(obj, to_float = True):
    if obj is not None:
        if isinstance(obj, np.ndarray):
            obj = torch.from_numpy(obj)
        if to_float:
            obj = obj.float()
        else:
            obj = obj.long()
        return Variable(obj) if type(obj) != Variable else obj
    return None