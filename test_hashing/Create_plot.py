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
    if title is not None:
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
        
    time_identification = np.array(time_identification)
    time_db = np.array(time_db)
        
    _, M = time_identification.shape
    
    time_average = np.mean(time_identification, axis=1)
    sorting = np.argsort(-time_average) # sort in decreasing order
    time_average = time_average[sorting]
    time_db = time_db[sorting]
    names = np.array(labels)[sorting]
    time_average_str = [time.strftime('%M:%S', time.gmtime(a)) for a in time_average]
    time_db_str = [time.strftime('%M:%S', time.gmtime(a)) for a in time_db]

    y = np.arange(0, 2*len(names), 2)
    height = 0.8  

    plt.figure(figsize=[6.4*1.3, 4.8*1.3])
    rects1 = plt.barh(y-height/2, time_average, height, color='r')
    rects2 = plt.barh(y+height/2, time_db, height, color='b')
    plt.bar_label(rects1, labels=time_average_str, padding=3)
    plt.bar_label(rects2, labels=time_db_str, padding=3)
    plt.legend([f'Identification (mean\nover {M} runs)', 'Database creation'])
    plt.xlabel('Time [min:sec]')
    
    xlocs, _ = plt.xticks()
    xlabels = [time.strftime('%M:%S', time.gmtime(a)) for a in xlocs]
    
    plt.xticks(xlocs, xlabels)
    plt.xlim(right=1.08*np.max(time_average)) # to fit labels
    plt.yticks(y, names)
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
    
def _find_lowest_biggest_frequencies(frequencies, path_attacked_imgs, kind, N):
    """
    Function to find image number of least identified images and most wrongly
    identified images from a frequency dictionary.

    Parameters
    ----------
    frequencies : Dictionary
        Contains the frequency of correctly/wrongly identification for each
        image for each algorithm and each BER.
    path_attacked_imgs : String
        Path to the attacked images.
    kind : String
        The kind of analysis it is used for. Either `recall` or `fpr`.
    N : Int
        Number of least correctly recorgnized/most wrongly recognized images
        to consider.

    Returns
    -------
    ID : Numpy array
        The numbers of N least correctly recognized images or N most wrongly 
        recognized images (depending on kind).
    value : Numpy array
        The actual number of times the N images were identified.

    """
    
    keys = np.array(list(frequencies.keys()))
    
    (N_rows, N_cols) = frequencies[keys[0]].shape 
    
    # The number of the images as ints
    img_numbers = np.array([int(key.split('.')[0].replace('data', '')) for key in keys])
    
    # Reorder the keys according to the image number
    sorting = np.argsort(img_numbers)
    img_numbers = img_numbers[sorting]
    keys = keys[sorting]
    
    ID = np.zeros((N_rows, N_cols, N))
    value = np.zeros((N_rows, N_cols, N))
    
    for i in range(N_rows):
        for j in range(N_cols):
            total_identified = np.zeros(len(keys))
            for k, key in enumerate(keys):
                total_identified[k] = frequencies[key][i, j]
                
            indices = np.argsort(total_identified, kind='mergesort')
            if kind == 'recall':
                lowest = indices[0:N]
                ID[i,j,:] = img_numbers[lowest]
                value[i,j,:] = total_identified[lowest]
            elif kind == 'fpr':
                biggest = indices[-N:]
                ID[i,j,:] = img_numbers[biggest]
                value = total_identified[biggest]
            
    return ID, value

    
def frequency_pannels(frequencies, path_attacked_imgs, kind, algo_names, metrics,
                      N=None, save=False, filename=None):
    """
    Creates frequency pannels showing scatter plots for each algo/constant metric
    with the frequency of identification.

    Parameters
    ----------
    frequencies : Dictionary
        Contains the frequency of correctly/wrongly identification for each
        image for each algorithm and each BER.
    path_attacked_imgs : String
        Path to the attacked images.
    kind : String
        The kind of analysis it is used for. Either `recall` or `fpr`.
    algo_names : List of str
        The names of the algorithms used.
    metrics : List
        The values of the current metric. Either values for recall, or FPR.
    N : Int, optional
        Number of least correctly recorgnized/most wrongly recognized images
        to consider. The default is None.
    save : Boolean, optional
        Whether to save the plots or not. The default is False.
    filename : String, optional
        Filename to save the plots. The default is None.

    Returns
    -------
    None.

    """
    
    if N is None and kind == 'recall':
        N = 20
    elif N is None and kind == 'fpr':
        N = 50
    
    N_rows = len(algo_names)
    N_cols = len(metrics)
    
    ID, value = _find_lowest_biggest_frequencies(frequencies, 
                                                 path_attacked_imgs, kind, N)
            
    if kind == 'recall':
        title = f'{N} least recognized images in correctly identified images (TP)'
        cols = [f'Recall {metrics[i]:.2f}' for i in range(len(metrics))]
        if save and filename is None:
            filename = 'Results/Mapping/pannel_TP.pdf'
    elif kind == 'fpr':
        title = f'{N} most recognized images in incorrectly identified images (FP)'
        cols = [f'FPR {metrics[i]:.2f}' for i in range(len(metrics))]
        if save and filename is None:
            filename = 'Results/Mapping/pannel_FP.pdf'
            
    pad = 5 # in points
        
    fig, axes = plt.subplots(N_rows, N_cols, figsize=(4*N_rows,3*N_cols), sharex=True,
                                 sharey='col')

    for i in range(N_rows):
        for j in range(N_cols):
            ax = axes[i][j]
            ax.scatter(ID[i,j,:], value[i,j,:], color='b', s=10)

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=20, ha='center', va='baseline')

    for ax, row in zip(axes[:,0], algo_names):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=20, ha='right', va='center')
            
    plt.suptitle(title, y=1, fontsize=30)
    fig.tight_layout()
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
    

def similarity_heatmaps(frequencies, path_attacked_imgs, kind, algo_names, metrics,
                      N=None, save=False, filename=None):
    """
    Creates similarity heatmaps plots for least correctly identified/most wrongly 
    identified images.

    Parameters
    ----------
    frequencies : Dictionary
        Contains the frequency of correctly/wrongly identification for each
        image for each algorithm and each BER.
    path_attacked_imgs : String
        Path to the attacked images.
    kind : String
        The kind of analysis it is used for. Either `recall` or `fpr`.
    algo_names : List of str
        The names of the algorithms used.
    metrics : List
        The values of the current metric. Either values for recall, or FPR.
    N : Int, optional
        Number of least correctly recorgnized/most wrongly recognized images
        to consider. The default is None.
    save : Boolean, optional
        Whether to save the plots or not. The default is False.
    filename : String, optional
        Filename to save the plots. The default is None.

    Returns
    -------
    None.

    """
    
    if N is None and kind == 'recall':
        N = 20
    elif N is None and kind == 'fpr':
        N = 50
    
    N_rows = len(algo_names)
    N_cols = len(metrics)
    
    ID, _ = _find_lowest_biggest_frequencies(frequencies, 
                                                 path_attacked_imgs, kind, N)
    
    if kind == 'recall':
        title = f'Similarity proportion between {N} least recognized images (TP)\n'
        if save and filename is None:
            filename = 'Results/Mapping/heatmap_TP_'
    elif kind == 'fpr':
        title = f'Similarity proportion between {N} most wrongly identified images (FP)\n'
        if save and filename is None:
            filename = 'Results/Mapping/heatmap_FP_'
            
    # Plot heatmaps with respect to algorithms
    for j in range(N_cols):
        similarities = np.zeros((N_rows, N_rows))
        
        for i in range(N_rows):
            for k in range(N_rows):
                similarities[i,k] = np.isin(ID[i,j,:], ID[k,j,:]).sum()/N
                
        frame = pd.DataFrame(similarities, columns=algo_names, index=algo_names)
        
        plt.figure()
        sns.heatmap(frame, center=0.5, annot=True)
        title_ =  title + kind + f' {metrics[j]:.2f}'
        plt.title(title_)
        if save:
            plt.savefig(filename + kind + f'_{metrics[j]:.2f}.pdf', bbox_inches='tight')
        plt.show()
        
    
    metrics_str = [kind + f' {metrics[i]:.2f}' for i in range(N_cols)]
        
    # Plot heatmaps with respect to metrics
    for i in range(N_rows):
        similarities = np.zeros((N_cols, N_cols))
        
        for j in range(N_cols):
            for k in range(N_cols):     
                similarities[j,k] = np.isin(ID[i,j,:], ID[i,k,:]).sum()/N
                
        frame = pd.DataFrame(similarities, columns=metrics_str, index=metrics_str)
        
        plt.figure()                
        sns.heatmap(frame, center=0.5, annot=True)  
        title_ = title + algo_names[i]
        plt.title(title_)
        if save:
            plt.savefig(filename + algo_names[i] + '.pdf', bbox_inches='tight')
        plt.show()
        
        
        