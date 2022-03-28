#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:24 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import generator
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import os
import hashing
import hashing.neuralhash as nh
from hashing.SimCLRv1 import resnet_wider
from hashing.SimCLRv2 import resnet as SIMv2
import scipy.spatial.distance as distance

#%%

image = Image.open('/Users/cyrilvallez/Desktop/Project/Datasets/BSDS500/Control/data221.jpg')

simclr = nh.load_simclr_v2(depth=101, width=2)(device='cpu')

transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
    T.CenterCrop(224),
    T.ToTensor()
    ])

tensor = torch.unsqueeze(transforms(image), dim=0)

out = simclr(tensor)


mem_params = sum([param.nelement()*param.element_size() for param in simclr.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in simclr.buffers()])
mem_ema = (mem_params + mem_bufs)/1e9 # in bytes

alloc = torch.cuda.max_memory_allocated()/1e9


#%%

path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_attacks/'
path_control = 'Datasets/BSDS500/Control_attacks/'
attacked = [path_experimental + file for file in os.listdir(path_experimental)[0:500]]
unknowns = [path_control + file for file in os.listdir(path_control)[0:500]]

algo = [
        hashing.NeuralAlgorithm('ResNet50 1x', raw_features=True, batch_size=150,
                        device='cpu', distance='L1')
        ]

db = hashing.create_databases(algo, path_database)
features = hashing.create_databases(algo, attacked)
unknown = hashing.create_databases(algo, unknowns)

db = list(db[0][0].values())
features = list(features[0][0].values())
unknown = list(unknown[0][0].values())

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

#%%
import os
from helpers import utils
from helpers import create_plot as plot

EXPERIMENT_NAME = 'Test_norms_thresholds/'

experiment_folder = 'Results/' + EXPERIMENT_NAME 
figure_folder = experiment_folder + 'Figures/'
   
if not os.path.exists(figure_folder + 'General/'):
    os.makedirs(figure_folder + 'General/')
if not os.path.exists(figure_folder + 'Attack_wise/'):
    os.makedirs(figure_folder + 'Attack_wise/')

general, attacks, _, _, global_time, db_time = utils.load_digest(experiment_folder)

algo_names = ['1', '2', '3', '4']
frame = plot.AUC_heatmap(attacks, algo_names=algo_names, save=True, filename='test')