import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success

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

def prediction(x):
    x_list = x[0].tolist()
    x_sort = sorted(x_list)
    max_index = x_list.index(x_sort[-1])

    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum

    return softmax, max_index#, sec_index

def AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, epsilon, pgd_alpha, fw, maxitr=50, max_eval=10000):
    ARTclassifier = PyTorchClassifier(
                model=targetmodel,
                clip_values=(0, 1),
                loss=F.cross_entropy,
                input_shape=(3, 32, 32),
                nb_classes=10,
            )
    L2_dist = []
    Attack = HopSkipJump(classifier=ARTclassifier, targeted =False, max_iter=maxitr, max_eval=max_eval)

    perturber = torch.load(f"materials/02_distillation/distill.pth", weights_only=False).cuda().eval()

    # Calculate results
    mid = int(len(data_loader.dataset)/2)
    member_groundtruth, non_member_groundtruth = [], []
    for idx, (data, target) in enumerate(data_loader):
        if idx >= mid:
            data, target = data.cuda(), target.cuda()
            if args.perturb_type == 'target':
                delta = emn(targetmodel, data, target, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            else:
                delta = emn(perturber, data, target, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            delta = delta.detach()
            data = torch.clamp(data + delta, min=lower_limit, max=upper_limit)
            data, target = data.cpu(), target.cpu()

        targetmodel.query_num = 0
        data = np.array(data)
        logit = ARTclassifier.predict(data)
        _, pred = prediction(logit)

        if pred != target.item():
            success = 1
            data_adv = data
        else:
            data_adv = Attack.generate(x=data) 
            data_adv = np.array(data_adv) 
            success = compute_success(ARTclassifier, data, [target.item()], data_adv)

        if success == 1:
            L2_dist.append(np.linalg.norm((data-data_adv).reshape((len(data),-1)), axis=1, ord=2))

            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)

    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L2_dist = np.asarray(L2_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)

    fw.write(f"ASR: {format(fpr[tpr>=0.9][0], '.3f')}.\n")
    print(f"ASR: {format(fpr[tpr>=0.9][0], '.3f')}.")
