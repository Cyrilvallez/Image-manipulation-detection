#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:07:37 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import os
import hashing
import hashing.neuralhash as nh
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

A = np.random.rand(4000)
B = np.random.rand(4000)


def jensen(a, b, base=2):
    
    A = torch.tensor(a)
    B = torch.tensor(b)
    
    A = A/torch.sum(A)
    B = B/torch.sum(B)
    
    M = (A+B)/2
    
    M = M.log()
    
    div = 1/2*(F.kl_div(M, A, reduction='sum') + F.kl_div(M, B, reduction='sum'))
        
    return torch.sqrt(div/np.log(base))

torch_res = jensen(A,B, base=2)
scipy_res = jensenshannon(A, B, base=2)

print(f'Torch : {torch_res:.3f}')
print(f'Scipy : {scipy_res:.3f}')

#%%

def jensen_test(A, B):
    A = torch.tensor(A)
    B = torch.tensor(B)
    out = torch.zeros((len(A), len(B)))
    
    A = A/torch.sum(A, axis=1)[:,None]
    B = B/torch.sum(B, axis=1)[:,None]

    for i,  feature in enumerate(B):
        C = torch.tile(feature, (len(A), 1))
        M = torch.log((A+C)/2)
        div = 1/2*(F.kl_div(M, A, reduction='none') + F.kl_div(M, C, reduction='none'))
                
        out[:, i] = torch.sum(div, axis=1)

    return out.numpy()


A = np.random.rand(250, 4000)
B = np.random.rand(1024, 4000)

test = jensen_test(A, B)

#%%

def jensen_test(A, B):
    with torch.no_grad():
        out = torch.zeros((len(A), len(B)))
        A = F.log_softmax(torch.from_numpy(A), dim=1)
        B = F.log_softmax(torch.from_numpy(B), dim=1)

        for i,  feature in enumerate(B):
            C = torch.tile(feature, (len(A), 1))
            M = (A+C)/2
            div = 1/2*(F.kl_div(M, A, reduction='sum', log_target=True) + \
                             F.kl_div(M, C, reduction='sum', log_target=True))
            print(div.shape)
                
            out[i, :] = div

    return out.numpy()

import sklearn.metrics as metrics

N = 1

res_loop = np.zeros((len(db), len(features)))

t0 = time.time()
for k in range(N):
    for i, feature_db in enumerate(db):
        for j, feature in enumerate(features):
            res_loop[i,j] = distance.jensenshannon(feature_db.features, feature.features)
            
dt_loop = (time.time() - t0)/N

A = np.zeros((len(db), len(db[0].features)))
B = np.zeros((len(features), len(features[0].features)))

for i, feature in enumerate(db):
    A[i,:] = feature.features
for i, feature in enumerate(features):
    B[i,:] = feature.features
    
t0 = time.time()
for k in range(N):
    res_scipy = distance.cdist(A,B,'jensenshannon', 2)
    
dt_scipy = (time.time() - t0)/N

t0 = time.time()
for k in range(N):
    res_torch = jensen_test(A,B)
    
dt_torch = (time.time() - t0)/N


#%%

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """

    return np.sum(np.where(p != 0, p * np.log2(p / q), 0))

def jensen_np(p, q):
    m = (p+q)/2

    return 1/2*(kl(p,m) + kl(q,m))

def jensen(vector, other):
    #A = F.log_softmax(vector, dim=0)
    #B = F.log_softmax(other, dim=0)
    
    A = torch.log(vector/vector.sum())
    B = torch.log(other/other.sum())
    
    M = (A+B)/2
    
    div = 1/2*(F.kl_div(M, A, reduction='sum', log_target=True) + \
                     F.kl_div(M, B, reduction='sum', log_target=True))
        
    return torch.sqrt(div)


def custom(c,d):
    
    #a = c/c.sum()
    #b = d/d.sum()
    a = F.softmax(c)
    b = F.softmax(d)
    
    m = (a+b)/2
    
    div = 1/2*(a*torch.log2(a/m) + b*torch.log2(b/m))
        
    return torch.sqrt(div.sum())
    
N = 100000
vector = torch.rand(6096)
other = torch.rand(6096)

t0 = time.time()
for k in range(N):
    res_scipy = distance.jensenshannon(vector, other, base=2)
    
dt_scipy = (time.time() - t0)/N

t0 = time.time()
for k in range(N):
    res_torch = jensen(vector, other)
    
dt_torch = (time.time() - t0)/N

t0 = time.time()
for k in range(N):
    res_custom = custom(vector, other)
    
dt_custom = (time.time() - t0)/N

print(f'scipy : {dt_scipy:.3e} s')
print(f'torch : {dt_torch:.3e} s')
print(f'custom : {dt_custom:.3e} s')
print(f'\nscipy : {res_scipy:.3e}')
print(f'torch : {res_torch:.3e}')
print(f'custom : {res_custom:.3e}')