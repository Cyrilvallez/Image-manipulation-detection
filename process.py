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

EXPERIMENT_NAME = 'Benchmark_norms_ImageNet/'

experiment_folder = 'Results/' + EXPERIMENT_NAME 
figure_folder = experiment_folder + 'Figures/'
   
if not os.path.exists(figure_folder + 'General/'):
    os.makedirs(figure_folder + 'General/')
if not os.path.exists(figure_folder + 'Attack_wise/'):
    os.makedirs(figure_folder + 'Attack_wise/')

general, attacks, _, _, global_time, db_time = utils.load_digest(experiment_folder)


#%%

save = True

if not os.path.exists(figure_folder + 'General/'):
    os.makedirs(figure_folder + 'General/')
if not os.path.exists(figure_folder + 'Attack_wise/'):
    os.makedirs(figure_folder + 'Attack_wise/')

a = plot.ROC_curves(general, save=save,
                filename=figure_folder + 'General/ROC_curves.pdf')
plot.ROC_curves(attacks, save=save,
                filename=figure_folder + 'Attack_wise/ROC')
plot.metrics_plot(general, save=save,
                  filename=figure_folder + 'General/Metrics')
plot.time_comparison(global_time, db_time, save=save,
                     filename=figure_folder + 'General/time.pdf')
plot.AUC_heatmap(attacks, save=save, filename=figure_folder + 'General/AUC')


#%%
#selected = ['Ahash 64 bits', 'Phash 64 bits', 'Dhash 64 bits', 'Whash 64 bits']
#subset = {key: value for key, value in general.items() if key in selected}

#plot.ROC_curves(subset)