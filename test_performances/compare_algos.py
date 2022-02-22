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
import matplotlib.pyplot as plt

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

save = True

# ROC curves
plt.figure(figsize=[6.4*1.5, 4.8*1.5])
for i in range(len(algos)):
    plt.plot(fpr[i,:], recall[i,:], '-+')
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (Recall)')
plt.legend(names)
plt.xticks(0.1*np.arange(11))
plt.yticks(0.1*np.arange(11))
plt.grid()
if save:
    plt.savefig('Results/General/Roc_curves.pdf', bbox_inches='tight')
plt.show()


# Accuracy, recall, precision curves
for i in range(len(algos)):
    name = names[i]
    plt.figure()
    plt.plot(BERs, accuracy[i,:], 'b-+')
    plt.plot(BERs, precision[i,:], 'r-+')
    plt.plot(BERs, recall[i,:], 'g-+')
    plt.plot(BERs, fpr[i,:], 'y-+')
    plt.xlabel('BER threshold')
    plt.ylabel('Metrics')
    plt.legend(['Accuracy', 'Precision', 'Recall', 'FPR'])
    plt.title(name)
    plt.grid()
    if save:
        plt.savefig('Results/General/Metrics_' + name + '.pdf', bbox_inches='tight')
    plt.show()
    
# Time bar plots
time_average = np.mean(time_identification, axis=1)
sorting = np.argsort(-time_average) # sort in decreasing order
time_average = time_average[sorting]
time_db = time_db[sorting]
names = np.array(names)[sorting]
time_average_str = [time.strftime('%M:%S', time.gmtime(a)) for a in time_average]
time_db_str = [time.strftime('%M:%S', time.gmtime(a)) for a in time_db]

y = np.arange(0, 2*len(names), 2)
height = 0.8  

max_ = int(np.max(np.floor(1/60*time_average)))
ticks = [f'{i*(max_//4)}:00' for i in range(6)]
x = [i*(max_//4)*60 for i in range(6)]

plt.figure(figsize=[6.4*1.3, 4.8*1.3])
rects1 = plt.barh(y-height/2, time_average, height, color='r')
rects2 = plt.barh(y+height/2, time_db, height, color='b')
plt.bar_label(rects1, labels=time_average_str, padding=3)
plt.bar_label(rects2, labels=time_db_str, padding=3)
plt.legend([f'Identification (mean\nover {len(BERs)} runs)', 'Database creation'])
plt.xlabel('Time [min:sec]')
plt.xticks(x, ticks)
plt.xlim(right=np.max(time_average) + 100) # to fit labels
plt.yticks(y, names)
if save:
    plt.savefig('Results/General/Time.pdf', bbox_inches='tight')
plt.show()

