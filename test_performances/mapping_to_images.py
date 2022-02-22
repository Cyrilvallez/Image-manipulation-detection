#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:16:55 2022

@author: cyrilvallez
"""

# Compares the different robust hash algorithms. Instead of just looking at
# binary classes identified/not identified, this script inspects which image
# map to which image in the database in order to detect potential difficulties
# of the algorithms


# IMPORTANT : This assumes that the attacked variations of the images CONTAIN
# the name of the original image to detect. For example, if the image is 
# road.jpg, the attacks must be e.g road_gaussian_noise_0.01.jpg 
# This is the case when using the default of the image generator

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from imagehash import imagehash as ih
from helpers import Plot
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

path_db = 'BSDS500/Identification/'
path_id = 'BSDS500/Identification_attacks/'
path_ct = 'BSDS500/Control_attacks/'

algos = [ih.average_hash, ih.phash, ih.dhash, ih.whash, ih.crop_resistant_hash]
names = ['average hash', 'phash', 'dhash', 'whash', 'crop resistant hash']
BERs = np.linspace(0, 0.2, 5)

# Initialize a dictionary of frequencies for each image
# Each key is an image name and its value is an array which will hold 
# the frequencies of detections for each algorithm and BER threshold
frequencies = {}

for file in os.listdir(path_db):
    # Convention : last dimension hold (correct identification, correct 
    # identification but to wrong image, incorrect identification)
    frequencies[file] = np.zeros((len(algos), len(BERs), 3)).astype(int)


for i in tqdm(range(len(algos))):
    
    algo = algos[i]
    name = names[i]
    
    # Create the database
    db = {}

    for file in os.listdir(path_db):
        img = Image.open(path_db + file)
        db[file] = algo(img)
        

    # Identification
    for j, rate in tqdm(enumerate(BERs)):
    
        for file in os.listdir(path_id):
            img = Image.open(path_id + file)
            hash_ = algo(img)
            detected = hash_.match_db_image(db, bit_error_rate=rate)
            for name in detected:
                # this was correctly identified. Assumes that the first '_' in 
                # the original file separates the name and the attack id
                if file.split('_')[0] in name:
                    frequencies[name][i,j,0] += 1
                # This was correctly identified but not for the right image in the db
                else:
                    frequencies[name][i,j,1] += 1
        
        for file in os.listdir(path_ct):
            img = Image.open(path_ct + file)
            hash_ = algo(img)
            detected = hash_.match_db_image(db, bit_error_rate=rate)
            for name in detected:
                # this was incorrectly identified
                frequencies[name][i,j,2] += 1

 
np.save('frequencies.npy', frequencies)       
        
#%%
# Plots

keys = np.array(list(frequencies.keys()))
index = np.arange(len(keys))
img_number = np.zeros(len(index))

for i, key in enumerate(keys):
    number = key.split('.')[0].replace('data', '')
    img_number[i] = int(number)
    
    
#%%

def plot_algo(row, dim, axs, N=5):
    tot = np.zeros((len(keys), N))
    for i, key in enumerate(keys):
        for j in range(N):
            tot[i,j] = frequencies[key][row,j,dim]
            
    for j, ax in enumerate(axs):
        valid = tot[:,j] > 0
        ax.scatter(index[valid], tot[:,j][valid])
        ax.set_xticks(index, img_number)
        
        
        
        
         
        
    
fig, axes = plt.subplots(7, 5, figsize=(20,15), sharex=True, sharey='col')

for i in range(len(algos)):
    plot_algo(i, 0, axes[i])
    
    
cols = [f'BER {BERs[j]:.2f}' for j in range(5)]
rows = np.array(names)
pad = 5 # in points

for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=20, ha='center', va='baseline')

for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=20, ha='right', va='center')
    
fig.tight_layout()
#fig.subplots_adjust(left=0.15, top=0.95)
    
plt.savefig('test.pdf', bbox_inches='tight')
plt.show()

#%%

keys = list(frequencies.keys())
index = np.arange(250)
tot = np.zeros(250)

for i, key in enumerate(keys):
    tot[i] = frequencies[key][0,-1,1]
    
plt.figure()
plt.scatter(index[tot > 0], tot[tot > 0])
plt.show()

plt.figure()
plt.scatter(np.array(keys)[tot > 0], tot[tot > 0])
plt.xticks(np.array(keys)[tot > 0], np.array(keys)[tot > 0], rotation='vertical')
plt.show()


#%%
N = 0
N2 = 0
test = []
problem = []

file = os.listdir(path_id)[56]
for name in detected:
        # this was correctly identified, assumes that the only dot in 
        # the original name is for the extension
        test.append(name.split('.')[0])
        if file.split('_')[0] in name:
            N += 1
            problem.append(name)
        # This was correctly identified but not for the right image in the db
        else:
            N2 += 1