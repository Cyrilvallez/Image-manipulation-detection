#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 08:58:25 2022

@author: cyrilvallez
"""

# =============================================================================
# Find the BER threshold for a constant Recall for one algo
# =============================================================================

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from hashing import imagehash as ih
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

path_db = 'BSDS500/Identification/'
path_id = 'BSDS500/Identification_attacks/'
path_ct = 'BSDS500/Control_attacks/'

algo = ih.phash
BERs = np.linspace(0.31, 0.35, 6)

accuracy = np.zeros(len(BERs))
precision = np.zeros(len(BERs))
recall = np.zeros(len(BERs)) 
fpr = np.zeros(len(BERs))

#%%
    
# Create the database
db = []

for file in os.listdir(path_db):
    img = Image.open(path_db + file)
    db.append(algo(img))
    

# Identification
for i, rate in tqdm(enumerate(BERs)):
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    
    #for file in os.listdir(path_id):
    #    img = Image.open(path_id + file)
    #    hash_ = algo(img)
    #    res = hash_.match_db(db, bit_error_rate=rate)
    #    if res:
    #        TP += 1
    #    else:
    #        FN += 1
        
    for file in os.listdir(path_ct):
        img = Image.open(path_ct + file)
        hash_ = algo(img)
        res = hash_.match_db(db, bit_error_rate=rate)
        if res:
            FP += 1
        else:
            TN += 1
            
    #accuracy[i] = (TP + TN)/(TP + TN + FP + FN)
    #try:
    #    precision[i] = TP/(TP + FP)
    #except ZeroDivisionError:
    #    precision[i] = 0
    #recall[i] = TP/(TP + FN)
    fpr[i] = FP/(FP + TN)
        
#%%
# Plots

plt.figure()
plt.plot(BERs, fpr, 'g-+')
plt.xlabel('BER threshold')
plt.ylabel('FPR')
plt.grid()
plt.show()

