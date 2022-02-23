#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:36:05 2022

@author: cyrilvallez

"""

# Compares the different robust hash algorithms

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from imagehash import imagehash as ih
from helpers import Plot
from PIL import Image
import time
from tqdm import tqdm
import Create_plot as plot

path_db = 'BSDS500/Identification/'
path_id = 'BSDS500/Identification_attacks/'
path_ct = 'BSDS500/Control_attacks/'

algos = [ih.average_hash, ih.phash, ih.dhash, ih.whash, ih.colorhash, ih.hist_hash,
         ih.crop_resistant_hash]
names = ['average hash', 'phash', 'dhash', 'whash', 'color hash', 'hist hash',
         'crop resistant hash']
BERs = np.linspace(0, 0.4, 10)

time_db = np.zeros(len(algos))
time_identification = np.zeros((len(algos), len(BERs)))
accuracy = np.zeros((len(algos), len(BERs)))
precision = np.zeros((len(algos), len(BERs)))
recall = np.zeros((len(algos), len(BERs)))
fpr = np.zeros((len(algos), len(BERs)))

#%%

for i in tqdm(range(len(algos))):
    
    algo = algos[i]
    name = names[i]
    
    # Create the database
    db = []
    t0 = time.time()

    for file in os.listdir(path_db):
        img = Image.open(path_db + file)
        db.append(algo(img))
    
    time_db[i] = time.time() - t0

    # Identification
    for j, rate in tqdm(enumerate(BERs)):
    
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        t0 = time.time()
    
        for file in os.listdir(path_id):
            img = Image.open(path_id + file)
            hash_ = algo(img)
            res = hash_.match_db(db, bit_error_rate=rate)
            if res:
                TP += 1
            else:
                FN += 1
        
        for file in os.listdir(path_ct):
            img = Image.open(path_ct + file)
            hash_ = algo(img)
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
                filename='Results/General/Roc_curves.pdf')

filenames = ['Results/General/Metrics_' + name + '.pdf' for name in names]
plot.metrics_plot(accuracy, precision, recall, fpr, BERs, names, save=save,
                  filenames=filenames)

plot.time_comparison(time_identification, time_db, names, save=save,
                     filename='Results/General/Time.pdf')
    

