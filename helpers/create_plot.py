#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:22:28 2022

@author: cyrilvallez
"""

# =============================================================================
# Contains functions to create different plots
# =============================================================================

from helpers import configs_plot
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.ticker


def ROC_curves(digest, common_ticks=True, save=False, filename=None, legend=None,
               size_multiplier=1.5):
    """
    Plot ROC curves for each algorithm.

    Parameters
    ----------
    result_dic : Dictionary
        General or attack-wise digest of an experiment.
    common_ticks : Boolean, optional.
        Whether or not to set ticks from 0 to 1 with step 0.1 in both directions.
        The default is True.
    save : Boolean, optional
        Whether to save the figure or not. The default is False.
    filename : str, optional
        The filename used to save the file. The default is None.
    legend : list, optional
        An optional legend to override algorithm names.

    Raises
    ------
    ValueError
        If save is True but filename is None.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You must specify a filename to save the figure.')
        
    # Determine if the dictionary contains general or attack_wise results
    if 'accuracy' in list(list(digest.values())[0].values())[0].keys():
        attack_wise = False
    else:
        attack_wise = True
            
    # Initialize stuff
    if legend is None:
        infer_legend = True
        legend = []
    else:
        infer_legend = False
    
    if not attack_wise:
        fpr = [[] for i in digest.keys()]
        recall = [[] for i in digest.keys()]
    else:
        fpr = {}
        recall = {}
        for attack_name in list(list(digest.values())[0].values())[0].keys():
            fpr[attack_name] = [[] for i in digest.keys()]
            recall[attack_name] = [[] for i in digest.keys()]
        
    # Retrive the values as lists
    for i, algorithm in enumerate(digest.keys()):
        
        if infer_legend:
            legend.append(algorithm)
        
        # Sort according to thresholds value (for consistency)
        thresholds = np.array(list(digest[algorithm].keys()))
        thresholds_values = [float(threshold.rsplit(' ',1)[1]) for threshold in thresholds]
        sorting = np.argsort(thresholds_values)
        thresholds = thresholds[sorting]
        
        for threshold in thresholds:
            
            if not attack_wise:
                
                fpr[i].append(digest[algorithm][threshold]['fpr'])
                recall[i].append(digest[algorithm][threshold]['recall'])
            
            else:
                
                for attack_name in digest[algorithm][threshold].keys():
                    
                    fpr[attack_name][i].append(digest[algorithm][threshold][attack_name]['fpr'])
                    recall[attack_name][i].append(digest[algorithm][threshold][attack_name]['recall'])
                    
    # Plot the ROC curves          
    
    if not attack_wise:
    
        plt.figure(figsize=[6.4*size_multiplier, 4.8*size_multiplier])
        for i in range(len(fpr)):
            plt.plot(fpr[i], recall[i], '-+')
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (Recall)')
        plt.legend(legend)
        if common_ticks:
            plt.xlim([-0.03, 1.03])
            plt.ylim([0.37, 1.03])
            plt.xticks(0.2*np.arange(6))
            plt.yticks(0.1*np.arange(4, 11))
        plt.grid()
        if save:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        
    else:
        
        for attack in fpr.keys():
            
            plt.figure(figsize=[6.4*size_multiplier, 4.8*size_multiplier])
            for i in range(len(fpr[attack])):
                plt.plot(fpr[attack][i], recall[attack][i], '-+')
            plt.xlabel('False positive rate (FPR)')
            plt.ylabel('True positive rate (Recall)')
            plt.legend(legend)
                
            title = attack
            # for latex output (there is a '&' in some default attack names that latex
            # does not understand as litteral)
            if '&' in title:
                title = title.replace('&', '\\&')
            plt.title(title.replace('_', ' '))
                
            if common_ticks:
                plt.xlim([-0.01, 1.01])
                plt.ylim([0.39, 1.01])
                plt.xticks(0.2*np.arange(6))
                plt.yticks(0.1*np.arange(4, 11))
            plt.grid()
            if save:
                plt.savefig(filename + '_' + attack + '.pdf', bbox_inches='tight')
            plt.show()
    
    
    
def metrics_plot(general_digest, common_ticks=True, save=False, filename=None):
    """
    Creates metrics plot for different algorithms.

    Parameters
    ----------
    result_dic : Dictionary
        General digest of an experiment.
    common_ticks : Boolean, optional
        Whether to make all plots have the same ticks. The default is True.
    save : Boolean, optional
        Whether to save the figure or not. The default is False.
    filename : list of str of size N, optional
        The filenames used to save the files. The default is None.

    Raises
    ------
    ValueError
        If save is True but filename is None.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You must specify a filename to save the figure.')
        
    accuracy = [[] for i in general_digest.keys()]
    precision = [[] for i in general_digest.keys()]
    recall = [[] for i in general_digest.keys()]
    fpr = [[] for i in general_digest.keys()]
    thresholds = [[] for i in general_digest.keys()]
    names = []
    
    for i, algorithm in enumerate(general_digest.keys()):
        
        names.append(algorithm)
        
        # Sort according to thresholds value (for consistency)
        thresholds_ = np.array(list(general_digest[algorithm].keys()))
        thresholds_values = np.array([float(threshold.rsplit(' ',1)[1]) for threshold in\
                                      thresholds_])
        sorting = np.argsort(thresholds_values)
        thresholds_ = thresholds_[sorting]
        thresholds[i] = thresholds_values[sorting]
        
        for j, threshold in enumerate(thresholds_):
            
            accuracy[i].append(general_digest[algorithm][threshold]['accuracy'])
            precision[i].append(general_digest[algorithm][threshold]['precision'])
            recall[i].append(general_digest[algorithm][threshold]['recall'])
            fpr[i].append(general_digest[algorithm][threshold]['fpr'])
            
        
    for i in range(len(accuracy)):
        plt.figure()
        plt.plot(thresholds[i], accuracy[i], 'b-+')
        plt.plot(thresholds[i], precision[i], 'r-+')
        plt.plot(thresholds[i], recall[i], 'g-+')
        plt.plot(thresholds[i], fpr[i], 'y-+')
        plt.xlabel('Threshold')
        plt.ylabel('Metrics')
        plt.legend(['Accuracy', 'Precision', 'Recall', 'FPR'])
        plt.title(names[i])
        if common_ticks:
            plt.yticks(0.2*np.arange(6))
        plt.grid()
        if save:
            plt.savefig(filename + '_' + names[i] + '.pdf', bbox_inches='tight')
        plt.show()
        
        
        
def time_comparison(match_time_digest, db_time_digest, save=False, filename=None):
    """
    Creates a bar plot comparing the time needed for different algorithms.

    Parameters
    ----------
    match_time_digest : Dictionary
        Matching time digest of an experiment.
    db_time_digest : Dictionary
        Database creation time digest of an experiment.
    save : Boolean, optional
        Whether or not to save the plot. The default is False.
    filename : str, optional
        The filename used to save the file. The default is None.

    Raises
    ------
    ValueError
        If save is True but filename is None.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You must specify a filename to save the figure.')
        
    time_identification = []
    time_db = []
    labels = []
    
    assert(match_time_digest.keys() == db_time_digest.keys())
    
    for algorithm in match_time_digest.keys():
        time_identification.append(match_time_digest[algorithm])
        time_db.append(db_time_digest[algorithm])
        labels.append(algorithm)
        
    time_identification = np.array(time_identification)
    time_db = np.array(time_db)
        
    sorting = np.argsort(-time_identification) # sort in decreasing order
    time_identification = time_identification[sorting]
    time_db = time_db[sorting]
    names = np.array(labels)[sorting]
    time_identification_str = [time.strftime('%M:%S', time.gmtime(a)) for a in time_identification]
    time_db_str = [time.strftime('%M:%S', time.gmtime(a)) for a in time_db]

    y = np.arange(0, 2*len(names), 2)
    height = 0.8  

    plt.figure(figsize=[6.4*1.3, 4.8*1.3])
    rects1 = plt.barh(y-height/2, time_identification, height, color='r')
    rects2 = plt.barh(y+height/2, time_db, height, color='b')
    plt.bar_label(rects1, labels=time_identification_str, padding=3)
    plt.bar_label(rects2, labels=time_db_str, padding=3)
    plt.legend(['Hashing + Identification', 'Database creation'])
    plt.xlabel('Time [min:sec]')
    
    xlocs, _ = plt.xticks()
    xlabels = [time.strftime('%M:%S', time.gmtime(a)) for a in xlocs]
    
    plt.xticks(xlocs, xlabels)
    plt.xlim(right=1.08*np.max(time_identification)) # to fit labels
    plt.yticks(y, names)
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
def time_comparison_log(match_time_digest, db_time_digest, save=False, filename=None,
                         labels=None):
    """
    Creates a bar plot comparing the time needed for different algorithms.

    Parameters
    ----------
    match_time_digest : Dictionary
        Matching time digest of an experiment.
    db_time_digest : Dictionary
        Database creation time digest of an experiment.
    save : Boolean, optional
        Whether or not to save the plot. The default is False.
    filename : str, optional
        The filename used to save the file. The default is None.

    Raises
    ------
    ValueError
        If save is True but filename is None.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You must specify a filename to save the figure.')
        
    time_identification = []
    time_db = []
    
    assert(match_time_digest.keys() == db_time_digest.keys())
    
    for algorithm in match_time_digest.keys():
        time_identification.append(match_time_digest[algorithm])
        time_db.append(db_time_digest[algorithm])
        
    time_identification = np.array(time_identification)
    time_db = np.array(time_db)
        
    sorting = np.argsort(-time_identification) # sort in decreasing order
    time_identification = time_identification[sorting]
    time_db = time_db[sorting]
    names = np.array(labels)[sorting]
    time_identification_str = [time.strftime('%M:%S', time.gmtime(a)) for a in time_identification]

    y = np.arange(0, len(names), 1)
    height = 0.8  

    plt.figure(figsize=[6.4*1.1, 4.8*1.2])
    rects1 = plt.barh(y, time_identification + time_db)
    plt.bar_label(rects1, labels=time_identification_str, padding=3)
    plt.xlabel('Time [min:sec]')
    
    plt.xscale('log')
    
    xticks = np.array([10, 1*60, 2*60, 5*60, 15*60, 30*60])
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: time.strftime('%M:%S', time.gmtime(x))))
    
    plt.xlim(left=5, right=1.6*np.max(time_identification)) # to fit labels
    plt.yticks(y, names)
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
    
def _find_lowest_biggest_frequencies(image_wise_digest, kind, N):
    """
    Find image number of least identified images and most wrongly identified images
    from an image-wise digest from an experiment.

    Parameters
    ----------
    image_wise_digest : Dictionary
        Image-wise digest of an experiment.
    kind : String
        The kind of analysis it is used for. Either `constant recall` or 
        `constant fpr`.
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
    algo_names : List
        The names of the algorithm used.

    """
    
    keys = np.array(list(list(list(image_wise_digest.values())[0].values())[0].keys()))
    
    N_rows, N_cols = len(image_wise_digest.keys()), len(list(image_wise_digest.values())[0].keys())
    
    # The number of the images as randomly attributed ints
    img_numbers = np.arange(len(keys))
    
    # REORDER ??
    
    ID = np.zeros((N_rows, N_cols, N))
    value = np.zeros((N_rows, N_cols, N))
    algo_names = []
    
    # Retrive the values as lists
    for i, algorithm in enumerate(image_wise_digest.keys()):
    
        algo_names.append(algorithm)
        
        # Sort according to thresholds value (for consistency)
        thresholds = np.array(list(image_wise_digest[algorithm].keys()))
        thresholds_values = [float(threshold.rsplit(' ',1)[1]) for threshold in thresholds]
        sorting = np.argsort(thresholds_values)
        thresholds = thresholds[sorting]
    
        for j, threshold in enumerate(thresholds):
            
            total_identified = np.zeros(len(keys))
            
            for k, image_name in enumerate(image_wise_digest[algorithm][threshold].keys()):
                
                total_identified[k] = image_wise_digest[algorithm][threshold][image_name]
                
            indices = np.argsort(total_identified, kind='mergesort')
            if kind == 'constant recall':
                lowest = indices[0:N]
                ID[i,j,:] = img_numbers[lowest]
                value[i,j,:] = total_identified[lowest]
            elif kind == 'constant fpr':
                biggest = indices[-N:]
                ID[i,j,:] = img_numbers[biggest]
                value[i,j,:] = total_identified[biggest]
            
    return ID, value, algo_names

    
def frequency_pannels(image_wise_digest, kind, metrics, N=None, save=False,
                      filename=None):
    """
    Creates frequency pannels showing scatter plots for each algo/constant metric
    with the frequency of identification.

    Parameters
    ----------
    image_wise_digest : Dictionary
        Image-wise digest of an experiment.
    kind : String
        The kind of analysis it is used for. Either `constant recall` or 
        `constant fpr`.
    metrics : List
        The values of the current metric matching `kind`. Either values for recall,
        or FPR.
    N : Int
        Number of least correctly recorgnized/most wrongly recognized images
        to consider.
    save : Boolean, optional
        Whether to save the plots or not. The default is False.
    filename : String, optional
        Filename to save the plots. The default is None.

    Raises
    ------
    ValueError
        If save is True but filename is None.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You must specify a filename to save the figure.')
    
    assert (kind == 'constant recall' or kind == 'constant fpr')
    
    if N is None and kind == 'constant recall':
        N = 20
    elif N is None and kind == 'constant fpr':
        N = 50
    
    ID, value, algo_names = _find_lowest_biggest_frequencies(image_wise_digest,
                                                             kind, N)
    
    N_rows = len(algo_names)
    N_cols = len(metrics)
            
    if kind == 'constant recall':
        title = f'{N} least recognized images in correctly identified images (TP)'
        cols = [f'Recall {metrics[i]:.2f}' for i in range(len(metrics))]
        if save and filename is None:
            filename = 'Results/Mapping/pannel_TP.pdf'
    elif kind == 'constant fpr':
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
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad - 3, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=20, ha='right', va='center')
        ax.set(ylabel='Identification count')
        
    for ax in axes[-1]:
        ax.set(xlabel='Image number')

    plt.suptitle(title, y=1, fontsize=30)
    fig.tight_layout()
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    

def similarity_heatmaps(image_wise_digest, kind, metrics, N=None, save=False,
                        filename=None):
    """
    Creates similarity heatmaps plots for least correctly identified/most wrongly 
    identified images.

    Parameters
    ----------
    image_wise_digest : Dictionary
        Image-wise digest of an experiment.
    kind : String
        The kind of analysis it is used for. Either `constant recall` or 
        `constant fpr`.
    metrics : List
        The values of the current metric matching `kind`. Either values for recall,
        or FPR.
    N : Int
        Number of least correctly recorgnized/most wrongly recognized images
        to consider.
    save : Boolean, optional
        Whether to save the plots or not. The default is False.
    filename : String, optional
        Filename to save the plots. The default is None.

    Raises
    ------
    ValueError
        If save is True but filename is None.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You must specify a filename to save the figure.')
    
    assert (kind == 'constant recall' or kind == 'constant fpr')
    
    if N is None and kind == 'constant recall':
        N = 20
    elif N is None and kind == 'constant fpr':
        N = 50
    
    ID, _, algo_names = _find_lowest_biggest_frequencies(image_wise_digest,
                                                         kind, N)
    
    N_rows = len(algo_names)
    N_cols = len(metrics)
    
    if kind == 'constant recall':
        title = f'Similarity proportion between {N} least recognized images (TP)\n'
        if save and filename is None:
            filename = 'Results/Mapping/heatmap_TP_'
    elif kind == 'constant fpr':
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
            plt.savefig(filename + '_' + algo_names[i] + '.pdf', bbox_inches='tight')
        plt.show()
        
        
        
def AUC_heatmap(attacks_digest, algo_names=None, save=False, filename=None,
                legend=None):
    """
    Create a heatmap with Area Under ROC Curve (AUC) for attacks wise digest.

    Parameters
    ----------
    attacks_digest : Dictionary
        Attack-wise digest of an experiment.
    algo_names : List, optional
        A list of names to give to the algorithms. The default is None.
    save : Boolean, optional
        Whether to save the plots or not. The default is False.
    filename : String, optional
        Filename to save the plots. The default is None.

    Raises
    ------
    ValueError
        If save is True but filename is None.

    Returns
    -------
    None.

    """
    
    if save and filename is None:
        raise ValueError('You must specify a filename to save the figure.')
            
    # Initialize stuff
    if legend is None:
        infer_legend = True
        legend = []
    else:
        infer_legend = False
        
    fpr = {}
    recall = {}
    for attack_name in list(list(attacks_digest.values())[0].values())[0].keys():
        fpr[attack_name] = [[] for i in attacks_digest.keys()]
        recall[attack_name] = [[] for i in attacks_digest.keys()]
        
    # Retrive the values as lists
    for i, algorithm in enumerate(attacks_digest.keys()):
        
        if infer_legend:
            legend.append(algorithm)
        
        # Sort according to thresholds value (for consistency)
        thresholds = np.array(list(attacks_digest[algorithm].keys()))
        thresholds_values = [float(threshold.rsplit(' ',1)[1]) for threshold in thresholds]
        sorting = np.argsort(thresholds_values)
        thresholds = thresholds[sorting]
        
        for threshold in thresholds:
            
            for attack_name in attacks_digest[algorithm][threshold].keys():
                    
                fpr[attack_name][i].append(attacks_digest[algorithm][threshold][attack_name]['fpr'])
                recall[attack_name][i].append(attacks_digest[algorithm][threshold][attack_name]['recall'])
     
    AUC = {}
    for attack_name in fpr.keys():
        AUC[attack_name] = []
        
    # Compute area under the curve (AUC)
    for attack_name in fpr.keys():
        
        for i in range(len(fpr[attack_name])):
            
            auc = np.trapz(recall[attack_name][i], x=fpr[attack_name][i])
            AUC[attack_name].append(auc)
            
    if algo_names is None:
        frame = pd.DataFrame(AUC, index=legend)
    else:
        frame = pd.DataFrame(AUC, index=algo_names)
        
    # For latex rendering
    frame.rename(columns={'s&p_noise_0.15': 's\\&p_noise_0.15',
                          's&p_noise_0.05': 's\\&p_noise_0.05',
                          's&p_noise_0.1': 's\\&p_noise_0.1',
                          'cropping_60_and_rescaling': 'cropping_60',
                          'rotation_60_and_rescaling': 'rotation_60',
                          'color_enhancement_2': 'color_2',
                          'sharpness_enhancement_2': 'sharpness_2',
                          'contrast_enhancement_2': 'contrast_2',
                          'brightness_enhancement_2': 'brightness_2'
                          }, inplace=True)
    
    strong_attacks = [
        'gaussian_noise_0.05',
        'speckle_noise_0.05',
        's\\&p_noise_0.15',
        'gaussian_filter_7x7',
        'median_filter_7x7',
        'jpg_compression_10',
        'cropping_60',
        'rotation_60',
        'shearing_20',
        'scaling_1.6',
        'text_length_50',
        'color_2',
        'sharpness_2',
        'contrast_2',
        'brightness_2',
        ]
    
    frame = frame[strong_attacks]
    
    labels_attacks = [' '.join(name.split('_')) for name in strong_attacks]

    plt.figure(figsize=[6.4*2.5, 4.8*1.5])    
    
    sns.heatmap(frame[strong_attacks].T, annot=True, yticklabels=labels_attacks,
                     square=True) 
    
    horizontal_divider = [0, 6, 11, 15]
    vertical_divider = [0, 5, 9, len(frame)]
    
    for i in horizontal_divider:
        plt.axhline(i, color='black', linewidth=3)
        
    for i in vertical_divider:
        plt.axvline(i, color='black', linewidth=3)
    
    if save:
        plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.show()
    
    
def heatmap_comparison_feature(general, time_general, time_db, algo_names=None, save=False, filename=None):
    
    if save and filename is None:
        raise ValueError('You must specify a filename to save the figure.')
            
    # Initialize stuff   
    legend = []
    fpr = []
    recall = []
    time_tot = []
        
    # Retrive the values as lists
    for i, algorithm in enumerate(general.keys()):
        
        legend.append(algorithm)
            
        time_tot.append(time_general[algorithm] + time_db[algorithm])
        
        # Sort according to thresholds value (for consistency)
        thresholds = np.array(list(general[algorithm].keys()))
        thresholds_values = [float(threshold.rsplit(' ',1)[1]) for threshold in thresholds]
        sorting = np.argsort(thresholds_values)
        thresholds = thresholds[sorting]
        
        fpr_list = []
        recall_list = []
        
        for threshold in thresholds:
                    
            fpr_list.append(general[algorithm][threshold]['fpr'])
            recall_list.append(general[algorithm][threshold]['recall'])
            
        fpr.append(fpr_list)
        recall.append(recall_list)
     
    AUC = []
        
    # Compute area under the curve (AUC)
    for i in range(len(fpr)):
        
        auc = np.trapz(recall[i], x=fpr[i])
        AUC.append(auc)
        
    
    AUC_per_algo = {'SIFT': np.zeros(4), 'ORB': np.zeros(4), 'DAISY': np.zeros(4),
             'LATCH': np.zeros(4)}
    time_per_algo = {'SIFT': np.zeros(4), 'ORB': np.zeros(4), 'DAISY': np.zeros(4),
             'LATCH': np.zeros(4)}
    descriptors = ['30', '100', '200', '300']
    
    for i, name in enumerate(legend):
        algo, N, _ = name.rsplit(' ', 2)
        try:
            algo = algo.rsplit(' ', 1)[1]
        except:
            pass
        index = descriptors.index(N)
        AUC_per_algo[algo][index] = AUC[i]
        time_per_algo[algo][index] = time_tot[i]
    
    
    frame_AUC = pd.DataFrame(AUC_per_algo, index=descriptors)
    frame_time = pd.DataFrame(time_per_algo, index=descriptors)

    plt.figure()    
    sns.heatmap(frame_AUC, annot=True, square=True, fmt='.3f', cmap='Blues',
                cbar=False) 
    plt.xlabel('Algorithm')
    plt.ylabel('Number of keypoints')
    if save:
        plt.savefig(filename + '_AUC.pdf', bbox_inches='tight')
    plt.show()
    
    plt.figure()    
    sns.heatmap(frame_time, annot=True, square=True, fmt='.3f', cmap='Reds',
                cbar=False) 
    plt.xlabel('Algorithm')
    plt.ylabel('Number of keypoints')

    for t in plt.gca().texts: 
        t.set_text(time.strftime('%M:%S', time.gmtime(float(t.get_text()))))
    
    if save:
        plt.savefig(filename + '_time.pdf', bbox_inches='tight')
    plt.show()
    
    return frame_AUC, frame_time
    

    
        
        
        
        