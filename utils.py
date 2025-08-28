from PIL import Image
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps
import os
import shutil
import pandas as pd
# from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.legend

def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

def dataset_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = list(range(sum(lengths)))
    np.random.seed(1)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

##############################################
##############################################
##############################################
import torch.nn.init as init
import numpy as np

init_param = np.sqrt(2)
init_type = 'default'
def init_func(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            if init_type == 'normal':
                init.normal_(m.weight, 0.0, init_param)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=init_param)
            elif init_type == 'xavier_unif':
                init.xavier_uniform_(m.weight, gain=init_param)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
            elif init_type == 'kaiming_out':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_param)
            elif init_type == 'zero':
                init.zeros_(m.weight)
            elif init_type == 'one':
                init.ones_(m.weight)
            elif init_type == 'constant':
                init.constant_(m.weight, init_param)
            elif init_type == 'default':
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()
