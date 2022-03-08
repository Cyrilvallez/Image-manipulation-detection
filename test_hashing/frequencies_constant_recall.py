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
from hashing import imagehash as ih
from PIL import Image
from tqdm import tqdm
import Create_plot as plot
import json

path_db = 'BSDS500/Identification/'
path_id = 'BSDS500/Identification_attacks/'
path_ct = 'BSDS500/Control_attacks/'

#algos = [ih.average_hash, ih.phash, ih.dhash, ih.whash, ih.crop_resistant_hash]
#names = ['average hash', 'phash', 'dhash', 'whash', 'crop resistant hash']
recall = [0.8, 0.9]
# Manual values for constant recall for each algo
#BERs = [[0.115, 0.19], [0.17, 0.31], [0.15, 0.27], [0.12, 0.18], [0.07, 0.2]]

algos = [ih.average_hash, ih.phash, ih.dhash, ih.whash]
names = ['average hash', 'phash', 'dhash', 'whash']
BERs = [[0.115, 0.19], [0.17, 0.31], [0.15, 0.27], [0.12, 0.18]]

#%%

# Initialize a dictionary of frequencies for each image
# Each key is an image name and its value is an array which will hold 
# the frequencies of detections for each algorithm and recall
frequencies = {}

# Find images supposed to be identified and append them to the dictionary
for key in os.listdir(path_db):
     for file in os.listdir(path_id):
         if file.split('_', 1)[0] == key.rsplit('.', 1)[0]:
             # Convention : the dictionary hold an array (algorithm, recall)
             frequencies[key] = np.zeros((len(algos), len(recall))).astype(int)
             break


for i in tqdm(range(len(algos))):
    
    algo = algos[i]
    name = names[i]
    BER = BERs[i]
    
    # Create the database
    db = {}

    for file in os.listdir(path_db):
        img = Image.open(path_db + file)
        db[file] = algo(img)
        

    # Identification
    for j, rate in tqdm(enumerate(BER)):
    
        for file in os.listdir(path_id):
            img = Image.open(path_id + file)
            hash_ = algo(img)
            detected = hash_.match_db_image(db, bit_error_rate=rate)
            for name in detected:
                # this was correctly identified. Assumes that the first '_' in 
                # the original file separates the name and the attack id 
                if file.split('_', 1)[0] == name.rsplit('.', 1)[0]:
                    frequencies[name][i,j] += 1
    
        
#%%
# Plots

save = True

plot.frequency_pannels(frequencies, path_id, 'recall', names, recall, save=save,
                       filename='Results/Mapping/test_pannel_')

plot.similarity_heatmaps(frequencies, path_id, 'recall', names, recall, save=save,
                       filename='Results/Mapping/test_heatmap_')

#%%

save = False
filename = 'TP_frequency_test.json' 

if save:
    with open(filename, 'w') as fp:
        json.dump(frequencies, fp, indent=1, default=lambda x: x.tolist())
        