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

#%%

import generator

path = 'Datasets/ILSVRC2012_img_val/Experimental/ILSVRC2012_val_00006778.JPEG'

a = generator.text_attack(path)

#%%
import generator
import time
import string
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os
from tqdm import tqdm

def text_attack(path, text_lengths=(10, 20, 30, 40, 50), **kwargs):
    """
    Generates random text at random position in the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    text_lengths : Tuple, optional
        Length (number of character) of the added text. The default
        is (10, 20, 30, 40, 50).

    Returns
    -------
    out : Dictionary
        Text added image (as PIL images).

    """
    
    width, height = image.size
    
    # List of characters for random strings
    characters = list(string.ascii_uppercase)
    # Get a font. The size is calculated so that it is 40 for a 512 by 512
    # image, and changes linearly from this reference, so it always
    # takes the same relative space on different images
    if min(width, height) >= 100:
        size = round(40*width/512)
    else:
        size = round(6*width/100)
        
    if width/height <= 0.5 or  width/height >= 2:
        
        if min(width, height) >= 100:
            size = round(40*(width+height)/2/512)
        else:
            size = round(6*(width+height)/2/100)
            
        size -= 2
    
    #if size < 7:
    #    size = round(40*(width+height)/2/512)
    #print(size)
    #size = int(np.floor(40*(width/512)))
    font = ImageFont.truetype('generator/Impact.ttf', size)
    
    out = {}

    for length in text_lengths:
        
        img = image.copy()
        # get a drawing context
        context = ImageDraw.Draw(img)
        
        # Random string generation
        sentence = ''.join(np.random.choice(characters, size=length,
                                            replace=True))
        
        # insert a newline every 20 characters for a 512-
        a = 20
        if height >= 2*width:
            a = int(np.floor(20*width/height))
        sentence = '\n'.join(sentence[i:i+a] for i in range(0, len(sentence), a))
        
        # Get the width and height of the text box
        dims = context.multiline_textbbox((0,0), sentence, font=font,
                                          stroke_width=2)
        width_text = dims[2] - dims[0]
        height_text = dims[3] - dims[1]
        
        # Random text position making sure that all text fits
        if (width-width_text-1 <= 1):
            print(f'width {width-width_text-5} image {width}x{height}')
            return
        if (height-height_text-1 <= 1):
            print(f'height {height-height_text-5} image {width}x{height}')
            return
        
        x = np.random.randint(1, width-width_text-1)
        y = np.random.randint(1, height-height_text-1)
        
        # Compute either white text black edegs or inverse based on mean
        #mean = np.mean(np.array(im)[x:x+w+1, y:y+h+1, :])
        #if mean <= 3*255/4:
        #    color = 'white'
        #    stroke = 'black'
        #else:
        #    color = 'black'
        #    stroke = 'white'
        
        color = 'white'
        stroke = 'black'

        context.multiline_text((x, y), sentence, font=font, fill=color,
                               stroke_fill=stroke, stroke_width=2, align='center')
        id_text = 'text_length_' + str(length)
        out[id_text] = img
        
    return out


"""
path = 'Datasets/ILSVRC2012_img_val/Experimental/'
imgs = [path + file for file in os.listdir(path)]
height = 50
width = 200
array = np.random.randint(0, 255, (height, width), dtype=np.uint8)
image = Image.fromarray(array)
res = text_attack(image)
res['text_length_50'].save('test3.pdf')
"""
#ratios = []

path = 'Datasets/ILSVRC2012_img_val/Experimental/'
imgs = [path + file for file in os.listdir(path)]

for img in tqdm(imgs):
    image = Image.open(img)
    foo = generator.perform_all_attacks(img, **generator.ATTACK_PARAMETERS)
    #foo = text_attack(image)
    #ratios.append((image.width,image.height))

      
#%%

height = 50
width = 50

size = 3

array = np.random.randint(0, 255, (height, width), dtype=np.uint8)
image = Image.fromarray(array)
characters = list(string.ascii_uppercase)
sentence = ''.join(np.random.choice(characters, size=50,
                                    replace=True))
# insert a newline every 20 characters for a 512-
a = int(np.floor(20*width/height))
if a > 25:
    a = 25
sentence = '\n'.join(sentence[i:i+a] for i in range(0, len(sentence), a))

context = ImageDraw.Draw(image)

font = ImageFont.truetype('generator/Impact.ttf', size)

# Get the width and height of the text box
dims = context.multiline_textbbox((0,0), sentence, font=font,
                                  stroke_width=2)
width_text = dims[2] - dims[0]
height_text = dims[3] - dims[1]

x = np.random.randint(1, width-width_text-1)
y = np.random.randint(1, height-height_text-1)

color = 'white'
stroke = 'black'

context.multiline_text((x, y), sentence, font=font, fill=color,
                       stroke_fill=stroke, stroke_width=2, align='center')

image.save('test4.png')

#%%

import cv2
from PIL import Image
import numpy as np

def DAISY(image, n_features=1):
    
    img = np.array(image.convert('L'))
    
    detector = cv2.ORB_create(nfeatures=n_features)
    extractor = cv2.xfeatures2d.DAISY_create()
    
    kps = detector.detect(img)
    _, descriptors = extractor.compute(img, kps)

    return descriptors

path = 'Datasets/ILSVRC2012_img_val/Experimental/ILSVRC2012_val_00006778.JPEG'
image = Image.open(path)

a = DAISY(image)

