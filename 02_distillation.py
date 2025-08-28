import os
import argparse
import logging
import sys
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn

from models import *
from utils import dataset_split

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--fname', default='02_distillation', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate (default: 0.001 for adam; 0.1 for SGD)')
    return parser.parse_args()
args = get_args()

def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = 0.95
    T = 6
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (T * T)

    return KD_loss

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

    epochs = 175

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    num_classes = 10
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=train_transform)

    train_size = 10000
    test_size = 5000
    distill_size = min(train_size, test_size)

    train_set, _ = dataset_split(train_dataset, [distill_size, len(train_dataset)-distill_size])
    test_set, _ = dataset_split(test_dataset, [distill_size, len(test_dataset)-distill_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model_distill = ResNet.ResNet18(num_classes=num_classes).cuda()
    
    params = model_distill.parameters()
    opt = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model_target = torch.load(f"materials/01_train_target_shadow/target.pth", weights_only=False).cuda().train()

    '''
    Training distill model
    '''
    fw.write("Training distill model.\n")

    for epoch in range(epochs):
        train_acc = 0
        train_loss = 0
        train_n = 0
        for (X, y) in test_loader:
            X, y = X.cuda(), y.cuda()

            output = model_distill(X)
            loss = loss_fn_kd(output, y, model_target(X))
    
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_acc += (output.max(1)[1]==y).sum().item()
            train_loss += loss*y.size(0)
            train_n += y.size(0)

        fw.write(f"Epoch {epoch}: train acc - {format(train_acc/train_n, '.3f')}; train loss - {format(train_loss/train_n, '.4f')}.\n")

    torch.save(model_distill, f"{fname}/distill.pth")

    '''
    Comparing distill and target model
    '''
    model_distill.eval()

    test_acc = 0
    test_n = 0
    for (X, y) in test_loader:
        X, y = X.cuda(), y.cuda()

        output = model_distill(X)

        test_acc += (output.max(1)[1]==y).sum().item()
        test_n += y.size(0)

    fw.write(f"Distill model on test set: test acc - {format(test_acc/test_n, '.3f')}.\n")

    test_acc = 0
    test_n = 0
    for (X, y) in test_loader:
        X, y = X.cuda(), y.cuda()

        output = model_target(X)

        test_acc += (output.max(1)[1]==y).sum().item()
        test_n += y.size(0)
    
    fw.write(f"Target model on test set: test acc - {format(test_acc/test_n, '.3f')}.\n")

if __name__ == "__main__":
    main()
