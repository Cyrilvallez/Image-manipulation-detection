#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:12:07 2022

@author: cyrilvallez
"""

# =============================================================================
# File to process and plot results from experiments
# =============================================================================

import os
from helpers import utils
from helpers import create_plot as plot

EXPERIMENT_NAME = 'test2/'

experiment_folder = 'Results/' + EXPERIMENT_NAME 
figure_folder = experiment_folder + 'Figures/'
   
if not os.path.exists(figure_folder + 'General/'):
    os.makedirs(figure_folder + 'General/')
if not os.path.exists(figure_folder + 'Attack_wise/'):
    os.makedirs(figure_folder + 'Attack_wise/')

general, attacks, _, _, global_time, db_time = utils.load_digest(experiment_folder)

#%%

a,b = plot.heatmap_comparison(general, global_time, db_time, save=True,
                              filename=figure_folder + 'General/comparison')


#%%

save = True

a = plot.ROC_curves(general, save=save,
                filename=figure_folder + 'General/ROC_curves.pdf')
# plot.ROC_curves(attacks, save=save,
                # filename=figure_folder + 'Attack_wise/ROC')
plot.metrics_plot(general, save=save,
                   filename=figure_folder + 'General/Metrics')
plot.time_comparison(global_time, db_time, save=save,
                       filename=figure_folder + 'General/time.pdf')
plot.AUC_heatmap(attacks, save=save, filename=figure_folder + 'General/AUC')


#%%
# selected = ['Ahash 64 bits', 'Phash 64 bits', 'Dhash 64 bits', 'Whash 64 bits', 
            # 'Crop resistant hash 64 bits']
# selected = ['SIFT 30 descriptors', 'ORB 30 descriptors', 'FAST + DAISY 30 descriptors',
            # 'FAST + LATCH 30 descriptors']
# selected = ['Inception v3 raw features Jensen-Shannon',
            # 'EfficientNet B7 raw features Jensen-Shannon',
            # 'ResNet50 2x raw features Jensen-Shannon',
            # 'ResNet101 2x raw features Jensen-Shannon',
            # 'SimCLR v1 ResNet50 2x raw features Jensen-Shannon',
            # 'SimCLR v2 ResNet50 2x raw features Jensen-Shannon',
            # 'SimCLR v2 ResNet101 2x raw features Jensen-Shannon',
            # ]

# legend = [name.split(' ', 1)[1] for name in selected]
legend = [name.split(' ', 1)[1] for name in general.keys()]
# for i in range(len(legend)):
    # if '+' in legend[i]:
        # legend[i] = legend[i].split('+ ', 1)[1]
# legend[-1] = 'Crop res'
# legend[0] = '*ResNet50 2x'
# legend[1] = '**ResNet50 2x'
# legend[2] = '**ResNet101 2x'

# subset = {key: value for key, value in general.items() if key in selected}

plot.ROC_curves(general, save=True, filename=figure_folder + 'General/ROC', legend=legend,
                size_multiplier=0.9, common_ticks=False)

#%%

names = []

for key in global_time.keys():
    if '64' in key:
        names.append(key.split(' 64', 1)[0])
    if '30' in key:
        names.append(key.split(' 30', 1)[0])
    if 'raw' in key:
        names.append(key.split(' raw', 1)[0])
        
for i in range(len(names)):
    if 'resistant' in names[i]:
        names[i] = 'Crop res'
    if 'LATCH' in names[i]:
        names[i] = 'LATCH'
    if 'DAISY' in names[i]:
        names[i] = 'DAISY'
    if 'SimCLR v1' in names[i]:
        names[i] = '*' + names[i].split('SimCLR v1 ', 1)[1]
    if 'SimCLR v2' in names[i]:
        names[i] = '**' + names[i].split('SimCLR v2 ', 1)[1]

plot.time_comparison_log(global_time, db_time, save=save,
                     filename=figure_folder + 'General/time.pdf', labels=names)

#%%
"""
from helpers import utils

name1 = 'Results/Benchmark_neural1_ImageNet'
name2 = 'Results/Benchmark_neural2_ImageNet'
name3 = 'Results/Benchmark_neural3_ImageNet'

digest1 = utils.load_digest(name1)
digest2 = utils.load_digest(name2)
digest3 = utils.load_digest(name3)

digests = [digest1, digest2, digest3]

big_digest = utils.merge_digests(digests)

utils.save_digest(big_digest, 'Results/Benchmark_neural_ImageNet')

"""


