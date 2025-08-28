import os
import argparse
import logging
import sys
import time
import math
import random

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset
import torch.backends.cudnn as cudnn

from models import *
from utils import dataset_split

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--fname', default='03_topconf_normal', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate (default: 0.001 for adam; 0.1 for SGD)')
    return parser.parse_args()
args = get_args()

def main():
    fname = f"materials/{args.fname}"
    if not os.path.exists(fname):
        os.makedirs(fname)

    fw = open(f"{fname}/output.log", "w")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True

    epochs = 200

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
    
    # Re-label members as 1 and non-members as 0
    train_dataset.targets[:] = [1] * len(train_dataset)
    test_dataset.targets[:] = [0] * len(test_dataset)

    mlp_size = 1000
    attack_target_pos, _ = dataset_split(train_dataset, [mlp_size, len(train_dataset)-mlp_size]) # Members to infer
    attack_target_neg, _ = dataset_split(test_dataset, [mlp_size, len(test_dataset)-mlp_size]) # Non-members to infer

    attack_target_set = ConcatDataset([attack_target_pos, attack_target_neg])
    attack_target_loader = torch.utils.data.DataLoader(attack_target_set, batch_size=1, shuffle=True)

    model_target = torch.load(f"materials/01_train_target_shadow/target.pth", weights_only=False).cuda().eval()

    # Calculate AUC
    groundtruth = []
    pred_prob = []
    for (X, y) in attack_target_loader:
        X, y = X.cuda(), y.cuda()

        pred = F.softmax(model_target(X), dim=1)
        top1 = torch.topk(pred, k=1, dim=1)[0].item()

        groundtruth.append(int(y))
        pred_prob.append(top1)

    groundtruth = np.array(groundtruth)
    pred_prob = np.array(pred_prob)

    fpr, tpr, _ = roc_curve(groundtruth, pred_prob, pos_label=1, drop_intermediate=False)

    fw.write(f"ASR: {format(fpr[tpr>=0.9][0], '.3f')}.\n")
    print(f"ASR: {format(fpr[tpr>=0.9][0], '.3f')}.")

if __name__ == "__main__":
    main()
