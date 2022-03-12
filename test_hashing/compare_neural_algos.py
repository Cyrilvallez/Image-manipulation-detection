#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:07:33 2022

@author: cyrilvallez
"""

# =============================================================================
# Compares the different neural robust hash algorithms
# =============================================================================

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from hashing import neuralhash as nh
from PIL import Image
import time
from tqdm import tqdm
import pandas as pd
import Create_plot as plot

path_db = 'BSDS500/Identification/'
path_id = 'BSDS500/Identification_attacks/'
path_ct = 'BSDS500/Control_attacks/'

algos = [nh.simclr_hash, nh.simclr_features]
names = ['SimCLR hash', 'SimCLR CS']
BERs = np.linspace(0, 0.4, 10)

time_db = np.zeros(len(algos))
time_identification = np.zeros((len(algos), len(BERs)))
accuracy = np.zeros((len(algos), len(BERs)))
precision = np.zeros((len(algos), len(BERs)))
recall = np.zeros((len(algos), len(BERs)))
fpr = np.zeros((len(algos), len(BERs)))

device = 'cuda'
batch_size = 256

#%%

for i in tqdm(range(len(algos))):
    
    algo = algos[i]
    name = names[i]
    
    # Create the database
    t0 = time.time()
    db = algo(path_db, batch_size=batch_size, device=device)
    time_db[i] = time.time() - t0

    # Identification
    for j, rate in tqdm(enumerate(BERs)):
    
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        t0 = time.time()
        
        hashes = algo(path_id, batch_size=batch_size, device=device)
        
        for hash_ in hashes:
            
            res = hash_.match_db(db, bit_error_rate=rate)
            if res:
                TP += 1
            else:
                FN += 1
    
        hashes = algo(path_ct, batch_size=batch_size, device=device)
        
        for hash_ in hashes:
            
            res = hash_.match_db(db, bit_error_rate=rate)
            if res:
                FP += 1
            else:
                TN += 1
            
        time_identification[i,j] = time.time() - t0
        accuracy[i,j] = (TP + TN)/(TP + TN + FP + FN)
        try:
            precision[i,j] = TP/(TP + FP)
        except ZeroDivisionError:
            precision[i,j] = 0
        recall[i,j] = TP/(TP + FN)
        fpr[i,j] = FP/(FP + TN)
        
#%%
# Plots

save=True

plot.ROC_curves(fpr, recall, names, save=save,
                filename='Results/General/ROC_curves_hash_feature.pdf')

filenames = ['Results/General/Metrics_' + name + '.pdf' for name in names]
plot.metrics_plot(accuracy, precision, recall, fpr, BERs, names, save=save,
                  filenames=filenames)

plot.time_comparison(time_identification, time_db, names, save=save,
                     filename='Results/General/Time_neural_hash_feature.pdf')
    
#%%
# Creates a pandas dataframe to easily save the data for later reuse

save_file=True

if save_file:
    frame = pd.DataFrame({
        'algo': names,
        'time_db': time_db,
        'time_identification': time_identification.tolist(),
        'accuracy': accuracy.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'fpr': fpr.tolist(),
        })
    
    frame.to_csv('Results/General/data_metrics_neural_hash_feature.csv', index=False)

#%%

data = pd.read_csv('Results/General/data_metrics_neural_hash_feature.csv')
names = data['algo']
time_db = pd.eval(data['time_db']).astype(float)
time_identification = pd.eval(data['time_identification']).astype(float)
accuracy = pd.eval(data['accuracy']).astype(float)
precision = pd.eval(data['precision']).astype(float)
recall = pd.eval(data['recall']).astype(float)
fpr = pd.eval(data['fpr']).astype(float)




