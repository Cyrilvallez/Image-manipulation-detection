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
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import Create_plot as plot
import pandas as pd
import seaborn as sns

path_db = 'BSDS500/Identification/'
path_id = 'BSDS500/Identification_attacks/'
path_ct = 'BSDS500/Control_attacks/'

algos = [ih.average_hash, ih.phash, ih.dhash, ih.whash, ih.crop_resistant_hash]
names = ['average hash', 'phash', 'dhash', 'whash', 'crop resistant hash']
BERs = np.linspace(0, 0.2, 5)

#%%

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
                if file.split('_', 1)[0] == name.rsplit('.', 1)[0]:
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

 
#%%
# Save the results
np.save('Results/Mapping/frequencies.npy', frequencies)       
        
#%%
# Plots
plot.frequency_pannels(frequencies, path_id, names, BERs, save=True,
                       filename='Results/Mapping/pannel_')

plot.similarity_heatmaps(frequencies, path_id, names, BERs, save=True,
                         filename='Results/Mapping/heatmap_')