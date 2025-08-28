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
from torchmetrics import ROC

from models import *
from utils import dataset_split

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--fname', default='03_shadmodel_normal', type=str)
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
    
    train_size = 10000
    mlp_size = 2000

    # Re-label members as 1 and non-members as 0
    train_dataset.targets[:] = [1] * len(train_dataset)
    test_dataset.targets[:] = [0] * len(test_dataset)

    train_target_set, train_shadow_set, _ = dataset_split(train_dataset, [train_size, train_size, len(train_dataset)-train_size-train_size])

    attack_target_pos, _ = dataset_split(train_target_set, [mlp_size, train_size-mlp_size])
    attack_shadow_pos, _ = dataset_split(train_shadow_set, [mlp_size, train_size-mlp_size])

    attack_target_neg, attack_shadow_neg, _ = dataset_split(test_dataset, [mlp_size, mlp_size, len(test_dataset)-mlp_size-mlp_size])

    attack_target_set = ConcatDataset([attack_target_pos, attack_target_neg])
    attack_shadow_set = ConcatDataset([attack_shadow_pos, attack_shadow_neg])

    attack_target_loader = torch.utils.data.DataLoader(attack_target_set, batch_size=args.batch_size, shuffle=False)
    attack_shadow_loader = torch.utils.data.DataLoader(attack_shadow_set, batch_size=args.batch_size, shuffle=True)

    model_target = torch.load(f"materials/01_train_target_shadow/target.pth", weights_only=False).cuda().eval()
    model_shadow = torch.load(f"materials/01_train_target_shadow/shadow.pth", weights_only=False).cuda().eval()

    if os.path.exists(f"{fname}/attack.pth"):
        model_attack = torch.load(f"{fname}/attack.pth", weights_only=False).cuda().eval()
    else:
        model_attack = Attack.MLP(3).cuda().train()

        params = model_attack.parameters()
        opt = torch.optim.Adam(params, lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        '''
        Training attack model
        '''
        fw.write("Training attack model.\n")

        for epoch in range(epochs):
            train_acc = 0
            train_loss = 0
            train_n = 0
            for (X, y) in attack_shadow_loader:
                X, y = X.cuda(), y.cuda()

                pred = F.softmax(model_shadow(X), dim=1).sort(descending=True, dim=1).values[:, :3]
                output = model_attack(pred)
                loss = criterion(output, y)
        
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_acc += (output.max(1)[1]==y).sum().item()
                train_loss += loss*y.size(0)
                train_n += y.size(0)

            fw.write(f"Epoch {epoch}: train advantage - {format((train_acc/train_n-0.5)*2, '.3f')}; train loss - {format(train_loss/train_n, '.4f')}.\n")

        model_attack.eval()
        test_acc = 0
        test_n = 0
        for (X, y) in attack_target_loader:
            X, y = X.cuda(), y.cuda()

            pred = F.softmax(model_target(X), dim=1).sort(descending=True, dim=1).values[:, :3]
            output = model_attack(pred)

            test_acc += (output.max(1)[1]==y).sum().item()
            test_n += y.size(0)

        fw.write(f"Epoch {epoch}: test advantage - {format((test_acc/test_n-0.5)*2, '.3f')}.\n")

        torch.save(model_attack, f"{fname}/attack.pth")

    # Calculate AUC
    attack_target_loader = torch.utils.data.DataLoader(attack_target_set, batch_size=1, shuffle=False)

    groundtruth = []
    pred_prob = []
    for (X, y) in attack_target_loader:
        X, y = X.cuda(), y.cuda()

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
