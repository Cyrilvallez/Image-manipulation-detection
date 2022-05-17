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

EXPERIMENT_NAME = 'Benchmark_general_BSDS500/'



if EXPERIMENT_NAME[-1] != '/':
    EXPERIMENT_NAME += '/'

experiment_folder = 'Results/' + EXPERIMENT_NAME 
figure_folder = experiment_folder + 'Figures/'
   
if not os.path.exists(figure_folder + 'General/'):
    os.makedirs(figure_folder + 'General/')
if not os.path.exists(figure_folder + 'Attack_wise/'):
    os.makedirs(figure_folder + 'Attack_wise/')

general, attacks, _, _, global_time, db_time = utils.load_digest(experiment_folder, True)


#%%

# For all plotting function, see helpers/create_plot.py

save = False

a = plot.ROC_curves(general, save=save,
                filename=figure_folder + 'General/ROC_curves.pdf')
# plot.ROC_curves(attacks, save=save,
                # filename=figure_folder + 'Attack_wise/ROC')
plot.metrics_plot(general, save=save,
                    filename=figure_folder + 'General/Metrics')
plot.time_comparison(global_time, db_time, save=save,
                        filename=figure_folder + 'General/time.pdf')
plot.AUC_heatmap(attacks, save=save, filename=figure_folder + 'General/AUC')


# To plot heatmaps comparing hash lengths
# this needs EXPERIMENT_NAME = 'Compare_hash_length_BSDS500/'
# plot.heatmap_comparison_classical(general, global_time, db_time)
                                  
# To plot heatmaps comparing number of keypoints
# this needs EXPERIMENT_NAME = 'Compare_N_keypoints_BSDS500/'
# plot.heatmap_comparison_feature(general, global_time, db_time)
                                   
# To plot heatmaps comparing distance metrics
# this needs EXPERIMENT_NAME = 'Compare_metrics_BSDS500/'
# plot.heatmap_comparison_neural(general, global_time, db_time)
                                    
# To plot heatmap comparing database size 
# digests = []
# db_size = [250, 2500, 25000]
# for size in db_size:
    # experiment_folder = f'Results/Database_{size}_ImageNet'
    # general, _, _, _, _, _ = utils.load_digest(experiment_folder, True)
    # general.append(digests)
# plot.heatmap_comparison_database(digests)




        
        


