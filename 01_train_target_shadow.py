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
    parser.add_argument('--fname', default='01_train_target_shadow', type=str)
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
    
    num_classes = 10
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    
    train_size = 10000
    test_size = 1000
    remain_size = len(train_dataset) - train_size - train_size - test_size
    train_target_set, train_shadow_set, _, test_set = dataset_split(train_dataset, [train_size, train_size, remain_size, test_size])

    train_target_loader = torch.utils.data.DataLoader(train_target_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    train_shadow_loader = torch.utils.data.DataLoader(train_shadow_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model_target = ResNet.ResNet18(num_classes=num_classes).cuda()
    model_shadow = ResNet.ResNet18(num_classes=num_classes).cuda()

    '''
    Training target model
    '''
    fw.write("Training target model.\n")

    params = model_target.parameters()
    opt = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model_target.train()
        train_acc = 0
        train_loss = 0
        train_n = 0
        for (X, y) in train_target_loader:
            X, y = X.cuda(), y.cuda()

            output = model_target(X)
            loss = criterion(output, y)
    
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_acc += (output.max(1)[1]==y).sum().item()
            train_loss += loss*y.size(0)
            train_n += y.size(0)

        fw.write(f"Epoch {epoch}: train acc - {format(train_acc/train_n, '.3f')}; train loss - {format(train_loss/train_n, '.4f')}.\n")

        model_target.eval()
        test_acc = 0
        test_n = 0
        for (X, y) in test_loader:
            X, y = X.cuda(), y.cuda()

            output = model_target(X)

            test_acc += (output.max(1)[1]==y).sum().item()
            test_n += y.size(0)

        fw.write(f"Epoch {epoch}: test acc - {format(test_acc/test_n, '.3f')}.\n")

    torch.save(model_target, f"{fname}/target.pth")

    '''
    Training shadow model
    '''
    fw.write("Training shadow model.\n")

    params = model_shadow.parameters()
    opt = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model_shadow.train()
        train_acc = 0
        train_loss = 0
        train_n = 0
        for (X, y) in train_shadow_loader:
            X, y = X.cuda(), y.cuda()

            output = model_shadow(X)
            loss = criterion(output, y)
    
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_acc += (output.max(1)[1]==y).sum().item()
            train_loss += loss*y.size(0)
            train_n += y.size(0)

        fw.write(f"Epoch {epoch}: train acc - {format(train_acc/train_n, '.3f')}; train loss - {format(train_loss/train_n, '.4f')}.\n")

        model_shadow.eval()
        test_acc = 0
        test_n = 0
        for (X, y) in test_loader:
            X, y = X.cuda(), y.cuda()

            output = model_shadow(X)

            test_acc += (output.max(1)[1]==y).sum().item()
            test_n += y.size(0)

        fw.write(f"Epoch {epoch}: test acc - {format(test_acc/test_n, '.3f')}.\n")

    torch.save(model_shadow, f"{fname}/shadow.pth")

if __name__ == "__main__":
    main()
