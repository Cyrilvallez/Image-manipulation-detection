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