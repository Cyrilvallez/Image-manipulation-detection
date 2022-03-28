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