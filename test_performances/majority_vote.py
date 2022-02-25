#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 22:42:01 2022

@author: cyrilvallez
"""

# =============================================================================
#  Compares the different algorithms with majority vote
# =============================================================================

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from imagehash import imagehash as ih
from generator import generate_attacks as ga
from PIL import Image
from tqdm import tqdm
import pandas as pd
import Create_plot as plot

# Parameters that were used for the attacks. This is needed to compute
# the ROC curves for each attack separately
GAUSSIAN_VARIANCES = (0.01, 0.02, 0.05)
SPECKLE_VARIANCES = (0.01, 0.02, 0.05)
SALT_PEPPER_AMOUNTS = (0.05, 0.1, 0.15)
GAUSSIAN_KERNELS = (1, 2, 3)
MEDIAN_KERNELS = (3, 5, 7)
COMPRESSION_QUALITY_FACTORS = (10, 50, 90)
SCALING_RATIOS = (0.4, 0.8, 1.2, 1.6)
CROPPING_PERCENTAGES = (5, 10, 20, 40, 60)
RESIZE_CROPPING = True
ROTATION_ANGLES = (5, 10, 20, 40, 60)
RESIZE_ROTATION = True
SHEARING_ANGLES = (1, 2, 5, 10, 20)
CONTRAST_FACTORS = (0.6, 0.8, 1.2, 1.4)
COLOR_FACTORS = (0.6, 0.8, 1.2, 1.4)
BRIGHTNESS_FACTORS = (0.6, 0.8, 1.2, 1.4)
SHARPNESS_FACTORS = (0.6, 0.8, 1.2, 1.4)
TEXT_LENGTHS = (10, 20, 30, 40, 50)

params = {
    'gaussian_variances': GAUSSIAN_VARIANCES,
    'speckle_variances': SPECKLE_VARIANCES,
    'salt_pepper_amounts': SALT_PEPPER_AMOUNTS,
    'gaussian_kernels': GAUSSIAN_KERNELS,
    'median_kernels': MEDIAN_KERNELS,
    'compression_quality_factors': COMPRESSION_QUALITY_FACTORS,
    'scaling_ratios': SCALING_RATIOS,
    'cropping_percentages': CROPPING_PERCENTAGES,
    'resize_cropping': RESIZE_CROPPING,
    'rotation_angles': ROTATION_ANGLES,
    'resize_rotation': RESIZE_ROTATION,
    'shearing_angles': SHEARING_ANGLES,
    'contrast_factors': CONTRAST_FACTORS,
    'color_factors': COLOR_FACTORS,
    'brightness_factors': BRIGHTNESS_FACTORS,
    'sharpness_factors': SHARPNESS_FACTORS,
    'text_lengths': TEXT_LENGTHS
    }

IDs = ga.retrieve_ids(**params)

path_db = 'BSDS5002/Identification/'
path_id = 'BSDS5002/Identification_attacks/'
path_ct = 'BSDS5002/Control_attacks/'
    
# Whether to save the plots on disk or not
save = True

# Initialize pandas dataframe to record the data and easily save it to file
frame = pd.DataFrame(columns=['attack', 'algos', 'fpr', 'recall', 'accuracy',
                              'precision'])

algos = [ih.average_hash, ih.phash, ih.dhash, ih.whash]
names = ['average hash', 'phash', 'dhash', 'whash', 'majority vote 2', 'majority vote 3']
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
    
    accuracy = np.zeros((len(names), len(BERs)))
    precision = np.zeros((len(names), len(BERs)))
    recall = np.zeros((len(names), len(BERs)))
    fpr = np.zeros((len(names), len(BERs)))

    # Loop over algos
    for i in range(len(names)):
        
        if (i<=3):
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
                if (i<=3):
                    hash_ = algo(img)
                    res = hash_.match_db(db, bit_error_rate=rate)
                    if res:
                        TP += 1
                    else:
                        FN += 1
                else:
                    hashes = [algo(img) for algo in algos]
                    res = np.array([hashes[i].match_db(DBs[i], bit_error_rate=rate) \
                                    for i in range(len(algos))])
                    vote = res.sum()
                    if (i == 4):
                        if (vote >= 2):
                            TP += 1
                        else:
                            FN += 1
                    elif (i == 5):
                        if (vote >= 3):
                            TP += 1
                        else:
                            FN += 1
                    
        
            for file in control:
                img = Image.open(path_ct + file)
                if (i<=3):
                    hash_ = algo(img)
                    res = hash_.match_db(db, bit_error_rate=rate)
                    if res:
                        FP += 1
                    else:
                        TN += 1
                else:
                    hashes = [algo(img) for algo in algos]
                    res = np.array([hashes[i].match_db(DBs[i], bit_error_rate=rate) \
                                    for i in range(len(algos))])
                    vote = res.sum()
                    if (i == 4):
                        if (vote >= 2):
                            FP += 1
                        else:
                            TN += 1
                    elif (i == 5):
                        if (vote >= 3):
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
            
    # Plot the ROC curves
    plot.ROC_curves(fpr, recall, names, title=' '.join(attack.split('_')),
                    save=save, large_ticks=False, 
                    filename='Results/Majority_vote/Roc_curves_' + attack + '.pdf')
    
    other = pd.DataFrame({
        'attack': attack, 
        'algos': [names], 
        'fpr': [fpr.tolist()],                          
        'recall': [recall.tolist()],
        'accuracy': [accuracy.tolist()],
        'precision': [precision.tolist()]
        })
    
    frame = pd.concat([frame, other], ignore_index=True)
    
#%%
# Save to files
frame.to_csv('Results/Majority_vote/data_metrics.csv', index=False)
            
    