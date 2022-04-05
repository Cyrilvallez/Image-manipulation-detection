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
import cupyx.scipy.special as special
import cupy as cp

#A = np.random.rand(4000)
#B = np.random.rand(4000)


def jensen(a, b, base=2):
    
    A = torch.tensor(a)
    B = torch.tensor(b)
    
    A = A/torch.sum(A)
    B = B/torch.sum(B)
    
    M = (A+B)/2
    
    M = M.log()
    
    div = 1/2*(F.kl_div(M, A, reduction='sum') + F.kl_div(M, B, reduction='sum'))
        
    return torch.sqrt(div/np.log(base))

def jensen_cu(a, b, base=2):
    
    A = a/a.sum()
    B = b/b.sum(axis=1)[:, None]
    print(B.shape[0])
    A = cp.tile(A, (B.shape[0], 1))
    
    M = (A+B)/2
    
    div = 1/2*(special.rel_entr(A, M).sum(axis=1) + special.rel_entr(B, M).sum(axis=1))
        
    return cp.sqrt(div/cp.log(base))


def jensen_array(a, B, base=2):
    
    a = torch.tensor(a).to('cuda')
    B = torch.tensor(B).to('cuda')
    
    a = a/torch.sum(a)
    B = B/torch.sum(B, axis=1)[:,None]
    A = torch.tile(a, (B.shape[0], 1))
    
    M = (a+B)/2
    
    M = M.log()
    
    div = 1/2*(F.kl_div(M, A, reduction='none').sum(dim=1) + F.kl_div(M, B, reduction='none').sum(dim=1))
        
    return np.array(torch.sqrt(div/np.log(base)).cpu())


#%%

from tqdm import tqdm 

a = np.random.rand(4000)
b = np.random.rand(10000, 4000)
N = 100

t0 = time.time()

for i in tqdm(range(N)):

    scipy_res = []
    for vec in b:
        scipy_res.append(jensenshannon(a, vec, base=2))
    scipy_res = np.array(scipy_res)
    
dt_scipy = (time.time() - t0)/N



t0 = time.time()

for i in tqdm(range(N)):

    torch_res = jensen_array(a,b, base=2)

dt_torch = (time.time() - t0)/N

print(f'Same : {(scipy_res == torch_res).all()}')
print('\n')
print(f'Scipy time : {dt_scipy:.2e}')
print(f'torch time : {dt_torch:.2e}')