#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:12:07 2022

@author: cyrilvallez
"""

# =============================================================================
# File to process and plot results from experiments
# =============================================================================

from helpers import utils
from helpers import create_plot as plot

experiment_folder = 'Results/First_benchmark/'
experiment_name = experiment_folder + 'First_benchmark'

experiment_results = experiment_folder + 'Figures/'

general, attacks, _, _, global_time, db_time = utils.load_digest(experiment_name)


#%%

save = True

plot.ROC_curves(general, save=save,
                filename=experiment_results + 'General/ROC_curves.pdf')
plot.ROC_curves(attacks, save=save,
                filename=experiment_results + 'Attack_wise/ROC')
plot.metrics_plot(general, save=save,
                  filename=experiment_results + 'General/Metrics')
plot.time_comparison(global_time, db_time, save=save,
                     filename=experiment_results + 'General/time.pdf')