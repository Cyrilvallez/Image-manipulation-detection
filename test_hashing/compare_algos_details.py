#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:19:25 2022

@author: cyrilvallez
"""

# =============================================================================
#  Compares the different robust hash algorithms in details for each
#  manipulation
# =============================================================================

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from hashing import imagehash as ih
import generator
from PIL import Image
from tqdm import tqdm
import pandas as pd
import Create_plot as plot



IDs = generator.retrieve_ids(**generator.ATTACK_PARAMETERS)

path_db = 'BSDS500/Identification/'
path_id = 'BSDS500/Identification_attacks/'
path_ct = 'BSDS500/Control_attacks/'
    
# Whether to save the plots on disk or not
save = True

# Initialize pandas dataframe to record the data and easily save it to file
frame = pd.DataFrame(columns=['attack', 'algos', 'fpr', 'recall', 'accuracy',
                              'precision'])

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
            
    # Plot the ROC curves
    plot.ROC_curves(fpr, recall, names, title=' '.join(attack.split('_')),
                    save=save, filename='Results/Details/Roc_curves_' + attack + '.pdf')
    
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
frame.to_csv('Results/Details/data_metrics.csv', index=False)
            
    