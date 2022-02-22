#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:19:25 2022

@author: cyrilvallez
"""

# Compares the different robust hash algorithms in details for each manipulation

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from imagehash import imagehash as ih
from generator import generate_attacks as ga
from helpers import Plot
from PIL import Image
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parameters that were used for the attacks. This is needed to compute
# the ROC curves for each attack separately
G_VAR = [0.01, 0.02, 0.05]
S_VAR = [0.01, 0.02, 0.05]
SP_AMOUNT = [0.05, 0.1, 0.15]
G_KERNEL = [1, 2, 3]
M_KERNEL = [3, 5, 7]
QUALITY_FACTORS = [10, 50, 90]
RATIOS = [0.4, 0.8, 1.2, 1.6]
PERCENTAGES = [5, 10, 20, 40, 60]
RESIZE_CROP = True
ANGLES_ROT = [5, 10, 20, 40, 60]
RESIZE_ROT = True
ANGLES_SHEAR = [1, 2, 5, 10, 20]
FACTORS_CONTRAST = [0.6, 0.8, 1.2, 1.4]
FACTORS_COLOR = [0.6, 0.8, 1.2, 1.4]
FACTORS_BRIGHT = [0.6, 0.8, 1.2, 1.4]
FACTORS_SHARP = [0.6, 0.8, 1.2, 1.4]
LENGTHS = [10, 20, 30, 40, 50]

params = {'g_var':G_VAR, 's_var':S_VAR, 'sp_amount':SP_AMOUNT, 'g_kernel':G_KERNEL,
          'm_kernel':M_KERNEL, 'quality_factors':QUALITY_FACTORS,
         'ratios':RATIOS, 'percentages':PERCENTAGES, 'resize_crop':RESIZE_CROP,
         'angles_rot':ANGLES_ROT, 'resize_rot':RESIZE_ROT, 'angles_shear':ANGLES_SHEAR,
         'factors_contrast':FACTORS_CONTRAST, 'factors_color':FACTORS_COLOR,
         'factors_bright':FACTORS_BRIGHT, 'factors_sharp':FACTORS_SHARP,
         'lengths':LENGTHS}

IDs = ga.retrieve_ids(**params)

path_db = 'BSDS500/Identification/'
path_id = 'BSDS500/Identification_attacks/'
path_ct = 'BSDS500/Control_attacks/'
    
# Whether to save the plots on disk or not
save = True

algos = [ih.average_hash, ih.phash, ih.dhash, ih.whash, ih.colorhash, ih.hist_hash,
         ih.crop_resistant_hash]
names = ['average hash', 'phash', 'dhash', 'whash', 'color hash', 'hist hash',
         'crop resistant hash']
BERs = np.linspace(0, 0.4, 10)

# First create each database
DBs = []
for algo in algos:
    db = []
    for file in os.listdir(path_db):
        img = Image.open(path_db + file)
        db.append(algo(img))
    DBs.append(db)
    
# Find all filenames for each attacks in each group
Identification = []
for attack in IDs:
    identifi = []
    for file in os.listdir(path_id):
        # Assumes that the filename convention is name_attackID.extension
        if attack == file.split('_', 1)[1].rsplit('.', 1)[0]:
            identifi.append(file)
    Identification.append(identifi)
    
Control = []
for attack in IDs:
    cont = []
    for file in os.listdir(path_ct):
        # Assumes that the filename convention is name_attackID.extension
        if attack == file.split('_', 1)[1].rsplit('.', 1)[0]:
            cont.append(file)
    Control.append(cont)

#%%
    
# Loop over all attacks and save ROC curves
for k, attack in tqdm(enumerate(IDs)):
    
    identification = Identification[k]
    control = Control[k]
    
    accuracy = np.zeros((len(algos), len(BERs)))
    precision = np.zeros((len(algos), len(BERs)))
    recall = np.zeros((len(algos), len(BERs)))
    fpr = np.zeros((len(algos), len(BERs)))

    # Loop over algos
    for i in range(len(algos)):
        
        algo = algos[i]
        db = DBs[i]
        
        # Identification
        for j, rate in enumerate(BERs):
    
            TP = 0
            FP = 0
            TN = 0
            FN = 0
    
            for file in identification:
                img = Image.open(path_id + file)
                hash_ = algo(img)
                res = hash_.match_db(db, bit_error_rate=rate)
                if res:
                    TP += 1
                else:
                    FN += 1
        
            for file in control:
                img = Image.open(path_ct + file)
                hash_ = algo(img)
                res = hash_.match_db(db, bit_error_rate=rate)
                if res:
                    FP += 1
                else:
                    TN += 1
                
            accuracy[i,j] = (TP + TN)/(TP + TN + FP + FN)
            try:
                precision[i,j] = TP/(TP + FP)
            except ZeroDivisionError:
                precision[i,j] = 0
            recall[i,j] = TP/(TP + FN)
            fpr[i,j] = FP/(FP + TN)
            
    # ROC curves
    plt.figure(figsize=[6.4*1.5, 4.8*1.5])
    for i in range(len(algos)):
        plt.plot(fpr[i,:], recall[i,:], '-+')
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (Recall)')
    plt.legend(names)
    plt.xticks(0.1*np.arange(11))
    plt.yticks(0.1*np.arange(11))
    title = ' '.join(attack.split('_'))
    # for latex output
    if '&' in title:
        title = title.replace('&', '\\&')
    plt.title(title)
    plt.grid()
    if save:
        plt.savefig('Results/Details/Roc_curves_' + attack + '.pdf', bbox_inches='tight')
    plt.show()
    