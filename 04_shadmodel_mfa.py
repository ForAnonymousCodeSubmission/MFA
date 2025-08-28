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
    parser.add_argument('--fname', default='04_shadmodel_mfa', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate (default: 0.001 for adam; 0.1 for SGD)')

    parser.add_argument('--attack-iters', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])

    parser.add_argument('--perturb-type', default='target', type=str, choices=['target', 'distill'])
    return parser.parse_args()
args = get_args()

upper_limit, lower_limit = 1,0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def emn(model, X, y, epsilon, alpha, attack_iters, restarts, norm, early_stop=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d - alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d - scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

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

    epochs = 200

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    epsilon = (8. / 255.)
    pgd_alpha = (epsilon / 10.)

    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)

    train_size = 10000
    mlp_size = 2000

    # Re-label members as 1 and non-members as 0
    for i in range(len(train_dataset)):
        train_dataset.targets[i] = (1, train_dataset.targets[i])
    for i in range(len(test_dataset)):
        test_dataset.targets[i] = (0, test_dataset.targets[i])

    train_target_set, train_shadow_set, _ = dataset_split(train_dataset, [train_size, train_size, len(train_dataset)-train_size-train_size])
    
    attack_target_pos, _ = dataset_split(train_target_set, [mlp_size, train_size-mlp_size]) # Members to infer
    attack_target_neg, _ = dataset_split(test_dataset, [mlp_size, len(test_dataset)-mlp_size]) # Non-members to infer and to test attack model

    attack_target_set = ConcatDataset([attack_target_pos, attack_target_neg])
    attack_target_loader = torch.utils.data.DataLoader(attack_target_set, batch_size=1, shuffle=False)

    model_target = torch.load(f"materials/01_train_target_shadow/target.pth", weights_only=False).cuda().eval()
    model_shadow = torch.load(f"materials/01_train_target_shadow/shadow.pth", weights_only=False).cuda().eval()

    perturber = torch.load(f"materials/02_distillation/distill.pth", weights_only=False).cuda().eval()
    model_attack = torch.load(f"materials/03_shadmodel_normal/attack.pth", weights_only=False).cuda()

    # Calculate AUC
    model_attack.eval()
    groundtruth = []
    pred_prob = []
    for (X, label) in attack_target_loader:
        X, (y, y_ori) = X.cuda(), label
        y, y_ori = y.cuda(), y_ori.cuda()

        if int(y[0]) == 0: # As batch_size=1 in this part
            if args.perturb_type == 'target':
                delta = emn(model_target, X, y_ori, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            else:
                delta = emn(perturber, X, y_ori, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            delta = delta.detach()
            X = torch.clamp(X + delta, min=lower_limit, max=upper_limit)

        # pred = F.sofmax(model_target(X), dim=1)
        pred = F.softmax(model_target(X), dim=1).sort(descending=True, dim=1).values[:, :3]
        output = model_attack(pred)

        groundtruth.append(int(y))
        pred_prob.append(float(F.softmax(output, dim=1)[:, 1]))

    groundtruth = np.array(groundtruth)
    pred_prob = np.array(pred_prob)

    fpr, tpr, _ = roc_curve(groundtruth, pred_prob, pos_label=1, drop_intermediate=False)
    
    fw.write(f"ASR: {format(fpr[tpr>=0.9][0], '.3f')}.\n")
    print(f"ASR: {format(fpr[tpr>=0.9][0], '.3f')}.")

if __name__ == "__main__":
    main()
