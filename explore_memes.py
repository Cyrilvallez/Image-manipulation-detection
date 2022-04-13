#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:56:42 2022

@author: cyrilvallez
"""

import os
import numpy as np
from helpers import utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

memes_path = 'Datasets/Kaggle_memes/Memes'
templates_path = 'Datasets/Kaggle_memes/Templates'

memes = [file.split('_', 1)[0] for file in os.listdir(memes_path)]
templates = [file.split('.', 1)[0] for file in os.listdir(templates_path)]

unique, counts = np.unique(memes, return_counts=True)

dic = utils.load_dictionary('Results/Test_ensemble/image_wise.json')

dic2 = dic[list(dic.keys())[0]]

count_algo = []

for img in unique:
    count_algo.append(dic2[img]['correct detection'])
    
tot_correct = sum([len(a['correct detection']) for a in list(dic2.values())])
tot_incorrect = sum([len(a['incorrect detection']) for a in list(dic2.values())])

all_detected = []
for key in dic2.keys():
    for name in dic2[key]['correct detection']:
        all_detected.append(key + '_' + name + '.jpg')
        
all_non_detected = np.setdiff1d(os.listdir(memes_path), all_detected)

"""
plt.figure()
plt.bar(x-0.2, counts, 0.4, color='b', label='True')
plt.bar(x+0.2, count_algo, 0.4, color='r', label='Detection')
plt.legend()
# plt.yscale('log')
"""

print(f'{tot_correct} / {np.sum(counts)} correct detections')
print(f'{tot_incorrect} incorrect detections')

#%%

tot_correct = np.zeros((len(list(dic.keys())), 20))
tot_incorrect = np.zeros(tot_correct.shape)

legend = []
for i, algo in enumerate(dic.keys()):
    legend.append(algo)
    for j, thresh in enumerate(dic[algo].keys()):
        tot_correct[i,j] = sum([a['correct detection'] for a in list(dic[algo][thresh].values())])
        tot_incorrect[i,j] = sum([a['incorrect detection'] for a in list(dic[algo][thresh].values())])
        
legend = [' '.join(a.split(' ', 3)[0:2]) for a in legend]
        
frame_correct = pd.DataFrame(tot_correct, index=legend)
frame_incorrect = pd.DataFrame(tot_incorrect, index=legend)
        
        
plt.figure(figsize=(20, 20))    
sns.heatmap(frame_correct, annot=True, square=True, fmt='g', cmap='Blues',
            cbar=False) 
plt.savefig('test_correct.pdf', bbox_inches='tight')
plt.show() 


plt.figure(figsize=(20, 20))    
sns.heatmap(frame_incorrect, annot=True, square=True, fmt='g', cmap='Reds',
            cbar=False) 
plt.savefig('test_incorrect.pdf', bbox_inches='tight')
plt.show()    
        
#%%

import glob
import shutil
import os

src_dir = 'Datasets/Kaggle_memes/Memes'
dst_dir = 'Datasets/Kaggle_memes/Non_detected'
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    shutil.copy(jpgfile, dst_dir)      
        