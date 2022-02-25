#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:22:28 2022

@author: cyrilvallez
"""

# =============================================================================
# Contains functions to create different plots
# =============================================================================

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from helpers import Plot
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import seaborn as sns


def ROC_curves(fpr, recall, legend, large_ticks=True, title=None,
               save=False, filename=None):
    """
    Plot different ROC curves on same figure for direct and efficient comparison.
    

    Parameters
    ----------
    fpr : array of shape (N, M)
        Each row contains the false positive rate data for a specific
        method (algorithm).
    recall : array of shape (N, M)
        Each row contains the recall data for a specific method (algorithm).
    legend : array of str of size N.
        Contains the str for the legend
    large_ticks : Boolean, optional.
        Whether or not to set ticks from 0 to 1 with step 0.1 in both directions.
        The default is True.
    title : str, optional
        Title to give to the figure. The default is None.
    save : Boolean, optional
        Whether to save the figure or not. The default is False.
    filename : str, optional
        The filename used to save the file. The default is None.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        filename = 'Results/General/ROC_curves.pdf'
        
    # for latex output (there is a '&' in some default attack names)
    if '&' in title:
        title = title.replace('&', '\\&')
    
    fpr = np.array(fpr)
    recall = np.array(recall)
    
    assert(fpr.shape == recall.shape)
    
    plt.figure(figsize=[6.4*1.5, 4.8*1.5])
    for i in range(len(fpr)):
        plt.plot(fpr[i,:], recall[i,:], '-+')
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (Recall)')
    plt.legend(legend)
    plt.title(title)
    if large_ticks:
        plt.xticks(0.1*np.arange(11))
        plt.yticks(0.1*np.arange(11))
    plt.grid()
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
def metrics_plot(accuracy, precision, recall, fpr, BERs, titles, common_ticks=True, 
                 save=False, filenames=None):
    """
    Creates metrics plot for different methods (algorithms).

    Parameters
    ----------
    accuracy : array of shape (N, M)
        Each row contains the accuracy data for a specific method (algorithm).
    precision : array of shape (N, M)
        Each row contains the precision data for a specific method (algorithm).
    recall : array of shape (N, M)
        Each row contains the recall data for a specific method (algorithm).
    fpr : array of shape (N, M)
        Each row contains the false positive rate data for a specific
        method (algorithm).
    BERs : array of shape (M)
        Contains the BERs thresholds for each data point.
    titles : array of str of size N
        Contains the name identifier for each method (algorithm) to use as title
    common_ticks : Boolean, optional
        Whether to make all plots have the same ticks. The default is True.
    save : Boolean, optional
        Whether to save the figure or not. The default is False.
    filenames : list of str of size N, optional
        The filenames used to save the files. The default is None.

    Returns
    -------
    None.

    """
    
    if save and filenames is None:
        filenames = [f'Results/General/Metrics_{titles[i]}.pdf' for i in range(len(accuracy))]
        
    accuracy = np.array(accuracy)
    precision = np.array(precision)
    recall = np.array(recall)
    fpr = np.array(fpr)
    
    assert((accuracy.shape == precision.shape) and (accuracy.shape == recall.shape) 
           and (accuracy.shape == fpr.shape))
        
    for i in range(len(accuracy)):
        plt.figure()
        plt.plot(BERs, accuracy[i,:], 'b-+')
        plt.plot(BERs, precision[i,:], 'r-+')
        plt.plot(BERs, recall[i,:], 'g-+')
        plt.plot(BERs, fpr[i,:], 'y-+')
        plt.xlabel('BER threshold')
        plt.ylabel('Metrics')
        plt.legend(['Accuracy', 'Precision', 'Recall', 'FPR'])
        plt.title(titles[i])
        if common_ticks:
            plt.yticks(0.2*np.arange(6))
        plt.grid()
        if save:
            plt.savefig(filenames[i], bbox_inches='tight')
        plt.show()
        
        
        
def time_comparison(time_identification, time_db, labels, save=False,
                    filename=None):
    """
    Creates a bar plot comparing the time needed for different methods

    Parameters
    ----------
    time_identification : array of shape (N, M)
        Each row containes the time needed for identification of all images
        for one method (algorithm).
    time_db : array of shape N
        Contains the time needed to create the database for all methods
        (algorithms).
    labels : array of str of shape N
        Contains the labels (names) for each method (algorithm).
    save : Boolean, optional
        Whether or not to save the plot. The default is False.
    filename : str, optional
        The filename used to save the file. The default is None.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        filename = 'Results/General/Time.pdf'
        
    _, M = time_identification.shape
    
    time_average = np.mean(time_identification, axis=1)
    sorting = np.argsort(-time_average) # sort in decreasing order
    time_average = time_average[sorting]
    time_DB = time_db[sorting]
    names = np.array(labels)[sorting]
    time_average_str = [time.strftime('%M:%S', time.gmtime(a)) for a in time_average]
    time_db_str = [time.strftime('%M:%S', time.gmtime(a)) for a in time_db]

    y = np.arange(0, 2*len(names), 2)
    height = 0.8  

    max_ = int(np.max(np.floor(1/60*time_average)))
    ticks = [f'{i*(max_//4)}:00' for i in range(6)]
    x = [i*(max_//4)*60 for i in range(6)]

    plt.figure(figsize=[6.4*1.3, 4.8*1.3])
    rects1 = plt.barh(y-height/2, time_average, height, color='r')
    rects2 = plt.barh(y+height/2, time_DB, height, color='b')
    plt.bar_label(rects1, labels=time_average_str, padding=3)
    plt.bar_label(rects2, labels=time_db_str, padding=3)
    plt.legend([f'Identification (mean\nover {M} runs)', 'Database creation'])
    plt.xlabel('Time [min:sec]')
    plt.xticks(x, ticks)
    plt.xlim(right=np.max(time_average) + 100) # to fit labels
    plt.yticks(y, names)
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
    
def _find_lowest_biggest_frequencies(frequencies, N_rows, N_cols,
                                     path_attacked_imgs, N_lowest, N_biggest):
    
    keys = np.array(list(frequencies.keys()))

    # Find images supposed to be identified 
    attacked_images = []
    for key in frequencies.keys():
        for file in os.listdir(path_attacked_imgs):
            if file.split('_', 1)[0] == key.rsplit('.', 1)[0]:
                attacked_images.append(key)
                break
    attacked_images = np.array(attacked_images)
    
    # The number of the images as ints
    img_numbers = np.array([int(key.split('.')[0].replace('data', '')) for key in keys])
    attacked_img_numbers = np.array([int(key.split('.')[0].replace('data', '')) for key in attacked_images])
        
    # Reorder the keys according to the image number
    sorting = np.argsort(img_numbers)
    img_numbers = img_numbers[sorting]
    keys = keys[sorting]
    
    sorting = np.argsort(attacked_img_numbers)
    attacked_img_numbers = attacked_img_numbers[sorting]
    attacked_images = attacked_images[sorting]
    
    ID_least_identified = np.zeros((N_rows, N_cols, N_lowest))
    ID_most_identified = np.zeros((N_rows, N_cols, N_biggest))
    value_least_identified = np.zeros((N_rows, N_cols, N_lowest))
    value_most_identified = np.zeros((N_rows, N_cols, N_biggest))
    
    for i in range(N_rows):
        for j in range(N_cols):
            tot_least_identified = []
            tot_most_identified = np.zeros(len(keys))
            for k, key in enumerate(keys):
                if key in attacked_images:
                    tot_least_identified.append(frequencies[key][i, j, 0])
                tot_most_identified[k] = frequencies[key][i, j, 1] + frequencies[key][i, j, 2]
            tot_least_identified = np.array(tot_least_identified)
            lowest = np.argsort(tot_least_identified, kind='mergesort')[0:N_lowest]
            biggest = np.argsort(tot_most_identified, kind='mergesort')[-N_biggest:]
            ID_least_identified[i,j,:] = attacked_img_numbers[lowest]
            ID_most_identified[i,j,:] = img_numbers[biggest]
            value_least_identified[i,j,:] = tot_least_identified[lowest]
            value_most_identified[i,j,:] = tot_most_identified[biggest]
            
    return (ID_least_identified, ID_most_identified, value_least_identified,
            value_most_identified)

    
def frequency_pannels(frequencies, path_attacked_imgs, algo_names, BERs, N_lowest=20,
                      N_biggest=50, save=False, filename=None):
    
    if save and filename is None:
        filename = 'Results/Mapping/pannel_'
    
    N_rows = len(algo_names)
    N_cols = len(BERs)
    
    (ID_least_identified, ID_most_identified, value_least_identified,
     value_most_identified) = _find_lowest_biggest_frequencies(frequencies, N_rows, N_cols,
                                          path_attacked_imgs, N_lowest, N_biggest)
            
    ID = [ID_least_identified, ID_most_identified]
    value = [value_least_identified, value_most_identified]
    title = [f'{N_lowest} least recognized images in correctly identified images',
             f'{N_biggest} most recognized images in incorrectly identified images']
    savenames = ['correct', 'incorrect']
    cols = [f'BER {BERs[i]:.2f}' for i in range(len(BERs))]
    rows = np.array(algo_names)
    pad = 5 # in points
            
    for k in range(2):
        
        X = ID[k]
        Y = value[k]
            
        fig, axes = plt.subplots(N_rows, N_cols, figsize=(20,15), sharex=True,
                                 sharey='col')

        for i in range(N_rows):
            for j in range(N_cols):
                ax = axes[i][j]
                ax.scatter(X[i,j,:], Y[i,j,:], color='b', s=10)

        for ax, col in zip(axes[0], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size=20, ha='center', va='baseline')

        for ax, row in zip(axes[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size=20, ha='right', va='center')
            
        plt.suptitle(title[k], y=1, fontsize=30)
        fig.tight_layout()
        if save:
            plt.savefig(filename + savenames[k] + '.pdf', bbox_inches='tight')
        plt.show()
    
    
    

def similarity_heatmaps(frequencies, path_attacked_imgs, algo_names, BERs,
                        N_lowest=20, N_biggest=50, save=False, filename=None):
    
    if save and filename is None:
        filename = 'Results/Mapping/heatmap_'
        
    N_rows = len(algo_names)
    N_cols = len(BERs)

    least_identified, most_identified, _, _ = _find_lowest_biggest_frequencies(
        frequencies, N_rows, N_cols, path_attacked_imgs, N_lowest, N_biggest)
            
    # Plot heatmaps with respect to algorithms
    for j in range(N_cols):
        similarities = np.zeros((N_rows, N_rows))
        
        for i in range(N_rows):
            for k in range(N_rows):
                similarities[i,k] = np.isin(least_identified[i,j,:],
                        least_identified[k,j,:]).sum()/N_lowest
                
        frame = pd.DataFrame(similarities, columns=algo_names, index=algo_names)
        
        plt.figure()
        sns.heatmap(frame, center=0.5, annot=True)
        title = f'Similarity proportion between {N_lowest} least recognized images'  + \
            f'\nBER threshold {BERs[j]:.2f}'
        plt.title(title)
        if save:
            plt.savefig(filename + f'{BERs[j]:.2f}.pdf', bbox_inches='tight')
        plt.show()
        
    
    BERs_str = [f'{BERs[i]:.2f}' for i in range(N_cols)]
        
    # Plot heatmaps with respect to BERs
    for i in range(N_rows):
        similarities = np.zeros((N_cols, N_cols))
        
        for j in range(N_cols):
            for k in range(N_cols):
                similarities[j,k] = np.isin(least_identified[i,j,:],
                        least_identified[i,k,:]).sum()/N_lowest
                
        frame = pd.DataFrame(similarities, columns=BERs_str, index=BERs_str)
        
        plt.figure()
        sns.heatmap(frame, center=0.5, annot=True)
        title = f'Similarity proportion between {N_lowest} least recognized images'  + \
            '\n' + algo_names[i]
        plt.title(title)
        if save:
            plt.savefig(filename + algo_names[i] + '.pdf', bbox_inches='tight')
        plt.show()
        
        
    # Plot heatmap with respect to algorithm for biggest non-detected
    similarities = np.zeros((N_rows, N_rows))
    
    for i in range(N_rows):
        for k in range(N_rows):
            similarities[i,k] = np.isin(most_identified[i,-1,:],
                most_identified[k,-1,:]).sum()/N_biggest
                
    frame = pd.DataFrame(similarities, columns=algo_names, index=algo_names)
        
    plt.figure()
    sns.heatmap(frame, center=0.5, annot=True)
    title = f'Similarity proportion between {N_biggest} most wrongly identified images'  + \
        f'\nBER threshold {BERs[-1]:.2f}'
    plt.title(title)
    if save:
        plt.savefig(filename + f'incorrect_{BERs[-1]:.2f}.pdf', bbox_inches='tight')
    plt.show()
        
        