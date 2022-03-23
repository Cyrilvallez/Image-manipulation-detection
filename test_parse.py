#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:32:41 2022

@author: cyrilvallez
"""
import argparse
import os
from helpers import utils

# Force the use of a user input path where to save the experiment at run-time,
# so that we do not mistakenly reuse the path from previous experiments
parser = argparse.ArgumentParser(description='Hashing experiment')
parser.add_argument('experiment_folder', type=str, help='A name for the experiment')
args = parser.parse_args()
experiment_folder = args.experiment_folder

results_folder = 'Results/'

# Check that it does not already exist
if experiment_folder in os.listdir(results_folder):
    raise ValueError('This experiment name is already taken. Choose another one.')
    
save_folder = results_folder + experiment_folder



dic = {'test': 4, 'test2':2}
foo = []

for i in range(6):
    foo.append(dic)
    
foo = tuple(foo)

experiment_folder = 'TEst/bbs/vvi'

utils.save_digest(foo, save_folder)
    