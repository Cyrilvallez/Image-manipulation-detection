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

experiment_folder = 'Results/Benchmark_dev_machine/'
figure_folder = experiment_folder + 'Figures/'

general, attacks, _, _, global_time, db_time = utils.load_digest(experiment_folder)


#%%

save = False

plot.ROC_curves(general, save=save,
                filename=figure_folder + 'General/ROC_curves.pdf')
plot.ROC_curves(attacks, save=save,
                filename=figure_folder + 'Attack_wise/ROC')
plot.metrics_plot(general, save=save,
                  filename=figure_folder + 'General/Metrics')
plot.time_comparison(global_time, db_time, save=save,
                     filename=figure_folder + 'General/time.pdf')