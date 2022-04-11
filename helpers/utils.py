#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:03:59 2022

@author: cyrilvallez
"""

# =============================================================================
# Contains some useful functions to process, save and load data
# =============================================================================

import json
import os
import argparse


def accuracy(TP, FN, FP, TN):
    return (TP + TN)/(TP + TN + FP + FN)


def precision(TP, FN, FP, TN):
    try:
        precision = TP/(TP + FP)
    except ZeroDivisionError:
        precision = 0
    return precision


def recall(TP, FN, FP, TN):
    return TP/(TP + FN)


def fpr(TP, FN, FP, TN):
    return FP/(FP + TN)


def save_dictionary(dictionary, filename):
    """
    Save a dictionary to disk as json file.

    Parameters
    ----------
    dictionary : Dictionary
        The dictionary to save.
    filename : str
        Filename to save the file.

    Returns
    -------
    None.

    """
    
    with open(filename, 'w') as fp:
        json.dump(dictionary, fp, indent='\t')
        
        
def load_dictionary(filename):
    """
    Load a json file and return a dictionary.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    data : Dictionary
        The dictionary representing the file.

    """
    
    file = open(filename)
    data = json.load(file)
    return data
    

DIGEST_FILE_NAMES = ('general.json', 'attacks.json', 'images_pos.json',
              'images_neg.json', 'match_time.json', 'db_time.json')


def save_digest(digest, experiment_folder):
    """
    Save a whole digest to disk, as returned by the `total_hashing` function.

    Parameters
    ----------
    digest : Tuple
        Tuple of dictionary representing an experiment results.
    experiment_folder : str
        A name for the files.

    Returns
    -------
    None.

    """
    
    # Make sure the path exists, and creates it if this is not the case
    exist = os.path.exists(experiment_folder)
    
    if not exist:
        os.makedirs(experiment_folder)
    
    if experiment_folder[-1] != '/':
        experiment_folder = experiment_folder + '/' 
    
    for dictionary, name in zip(digest, DIGEST_FILE_NAMES):
        save_dictionary(dictionary, experiment_folder + name)
        
        
def load_digest(experiment_folder):
    """
    Load the files corresponding to an experiment digest, as returned by the
    `total_hashing` function.

    Parameters
    ----------
    experiment_folder : str
        The name of the experiment.

    Returns
    -------
    Tuple
        The digest corresponding to the files.

    """
    
    if experiment_folder[-1] != '/':
        experiment_folder = experiment_folder + '/' 
        
    digest = []
    for name in DIGEST_FILE_NAMES:
        digest.append(load_dictionary(experiment_folder + name))
        
    return tuple(digest)


def merge_digests(digests):
    """
    Merge different similar digest into one.

    Parameters
    ----------
    digests : List
        List of tuples corresponding to the digests.

    Returns
    -------
    Tuple
        The digest corresponding to the merge of all others.

    """

    big_digest = []

    for i in range(len(digests[0])):
        
        dic = {}
        
        for j in range(len(digests)):

            dic = {**dic, **digests[j][i]}
            
        big_digest.append(dic)
        

    return tuple(big_digest)

                
def process_digests(positive_digest, negative_digest, attacked_image_names):
    """
    Process the digests from a hashing experiment with both an experimental and
    control group of images.

    Parameters
    ----------
    positive_digest : Tuple of dictionaries
        Digest from the experimental part of the experiment.
    negative_digest : Tuple of dictionaries
        Digest from the control part of the experiment.
    attacked_image_names : List
        Names of the attacked images in the experimental part of the experiment.

    Returns
    -------
    general_output : Dictionary
        Results at the global level (accuracy, precision, recall, fpr) for each algorith
        and threshold.
    attack_wise_output : Dictionary
        Results at the attack level (accuracy, precision, recall, fpr) for each algorith
        and threshold.
    image_wise_pos_output : Dictionary
        Results at the image level, for images supposed to be identified (number of 
        correct identification) for each algorith and threshold.
    image_wise_neg_output : Dictionary
        Results at the image level, for images not supposed to be identified (number of 
        incorrect identification) for each algorith and threshold.
    running_time : Dictionary
        Total running time (creating fingerprints and mean matching time over thresholds)
        for each algorithm.

    """
    
    general_pos, attack_wise_pos, image_wise_pos, running_time_pos = positive_digest
    general_neg, attack_wise_neg, image_wise_neg, running_time_neg = negative_digest
    
    general_output = {}
    attack_wise_output = {}
    image_wise_pos_output = {}
    image_wise_neg_output = {}
    running_time_output = {}
    
    for algorithm in general_pos.keys():
        
        general_output[algorithm] = {}
        attack_wise_output[algorithm] = {}
        image_wise_pos_output[algorithm] = {}
        image_wise_neg_output[algorithm] = {}
        running_time_output[algorithm] = running_time_pos[algorithm] + \
            running_time_neg[algorithm]
            
        for threshold in general_pos[algorithm].keys():
            
            attack_wise_output[algorithm][threshold] = {}
            image_wise_pos_output[algorithm][threshold] = {}
            image_wise_neg_output[algorithm][threshold] = {}
            
            TP = general_pos[algorithm][threshold]['detection']
            FN = general_pos[algorithm][threshold]['no detection']
            FP = general_neg[algorithm][threshold]['detection']
            TN = general_neg[algorithm][threshold]['no detection']
            
            general_output[algorithm][threshold] = {
                'accuracy': accuracy(TP, FN, FP, TN),
                'precision': precision(TP, FN, FP, TN),
                'recall': recall(TP, FN, FP, TN),
                'fpr': fpr(TP, FN, FP, TN)
                }
            
    
            for attack_name in attack_wise_pos[algorithm][threshold].keys():
                
                TP = attack_wise_pos[algorithm][threshold][attack_name]['detection']
                FN = attack_wise_pos[algorithm][threshold][attack_name]['no detection']
                FP = attack_wise_neg[algorithm][threshold][attack_name]['detection']
                TN = attack_wise_neg[algorithm][threshold][attack_name]['no detection']
                
                attack_wise_output[algorithm][threshold][attack_name] = {
                    'accuracy': accuracy(TP, FN, FP, TN),
                    'precision': precision(TP, FN, FP, TN),
                    'recall': recall(TP, FN, FP, TN),
                    'fpr': fpr(TP, FN, FP, TN)
                    }
                
            for image_name in image_wise_pos[algorithm][threshold].keys():
                
                image_wise_neg_output[algorithm][threshold][image_name] = \
                    image_wise_pos[algorithm][threshold][image_name]['incorrect detection'] + \
                        image_wise_neg[algorithm][threshold][image_name]['incorrect detection']
                
                if image_name in attacked_image_names:
                    
                    image_wise_pos_output[algorithm][threshold][image_name] = \
                        image_wise_pos[algorithm][threshold][image_name]['correct detection']
                        
    return (general_output, attack_wise_output, image_wise_pos_output, 
            image_wise_neg_output, running_time_output)


def parse_input():
    """
    Create a parser for command line arguments in order to get the experiment
    folder. Also check that this folder is valid, in the sense that it is not 
    already being used.
    
    Raises
    ------
    ValueError
        If the experiment name is already taken or not valid.

    Returns
    -------
    save_folder : str
        The path to the folder for saving the experiment digest.

    """
    # Force the use of a user input at run-time to specify the path 
    # so that we do not mistakenly reuse the path from previous experiments
    parser = argparse.ArgumentParser(description='Hashing experiment')
    parser.add_argument('experiment_folder', type=str, help='A name for the experiment')
    args = parser.parse_args()
    experiment_folder = args.experiment_folder

    if '/' in experiment_folder:
        raise ValueError('The experiment name must not be a path. Please provide a name without any \'/\'.')

    results_folder = 'Results/'
    save_folder = results_folder + experiment_folder 

    # Check that it does not already exist and contain results
    if experiment_folder in os.listdir(results_folder):
        if 'general.json' in os.listdir(save_folder):
            raise ValueError('This experiment name is already taken. Choose another one.')
            
    return save_folder
                        
            