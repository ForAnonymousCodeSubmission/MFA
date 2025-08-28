import os
import argparse
import logging
import sys
import time
import math
import random
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset
import torch.backends.cudnn as cudnn

from models import *
from mia_labelonly_hopskipjump_emn import AdversaryTwo_HopSkipJump
from utils import dataset_split

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--fname', default='04_labelonly_mfa', type=str)
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--attack-iters', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--norm', default='l_inf', type=str)

    parser.add_argument('--perturb-type', default='target', type=str, choices=['target', 'distill'])
    return parser.parse_args()
args = get_args()

def AdversaryTwo(data_loader, epsilon, pgd_alpha, fw):
    torch.cuda.empty_cache()
    
    targetmodel = torch.load(f"materials/01_train_target_shadow/target.pth", weights_only=False).cuda().eval()
    AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, epsilon, pgd_alpha, fw)
    
def main():
    fname = f"materials/{args.fname}"
    if not os.path.exists(fname):
        os.makedirs(fname)

    fw = open(f"{fname}/output_{args.perturb_type}.log", "w")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    epsilon = (8. / 255.)
    pgd_alpha = (epsilon / 10.)
    
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=train_transform)
    
    eval_size = 100
    mem_set, _ = dataset_split(train_dataset, [eval_size, len(train_dataset)-eval_size])
    non_set, _ = dataset_split(test_dataset, [eval_size, len(test_dataset)-eval_size])

    data_set = ConcatDataset([mem_set, non_set])
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)

    AdversaryTwo(data_loader, epsilon, pgd_alpha, fw)

if __name__ == "__main__":
    main()
