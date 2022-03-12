#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:10:03 2022

@author: cyrilvallez
"""

import numpy as np
from tqdm import tqdm
import time
import hashing.imagehash as ih
import hashing.neuralhash as nh

def general_performances(methods, thresholds, path_to_db, path_to_identification,
                         path_to_control):
    
    time_db = np.zeros(len(methods))
    names = np.zeros(len(methods)).astype(str)
    time_identification = np.zeros((len(methods), len(thresholds)))
    accuracy = np.zeros((len(methods), len(thresholds)))
    precision = np.zeros((len(methods), len(thresholds)))
    recall = np.zeros((len(methods), len(thresholds)))
    fpr = np.zeros((len(methods), len(thresholds)))

    for i in tqdm(range(len(methods))):
        
        method = methods[i]
        names[i] = str(method)
        
        # Create the database
        t0 = time.time()
        db = method(path_to_db)
        time_db[i] = time.time() - t0

        # Identification
        for j, threshold in tqdm(enumerate(thresholds)):
        
            TP = 0
            FP = 0
            TN = 0
            FN = 0

            t0 = time.time()
            
            fingerprints = method(path_to_identification)
            
            for fingerprint in fingerprints:

                res = fingerprint.match_db(db, threshold=threshold)
                if res:
                    TP += 1
                else:
                    FN += 1
            
            fingerprints = method(path_to_control)
            
            for fingerprint in fingerprints:

                res = fingerprint.match_db(db, threshold=threshold)
                if res:
                    FP += 1
                else:
                    TN += 1
                
            time_identification[i,j] = time.time() - t0
            accuracy[i,j] = (TP + TN)/(TP + TN + FP + FN)
            try:
                precision[i,j] = TP/(TP + FP)
            except ZeroDivisionError:
                precision[i,j] = 0
            recall[i,j] = TP/(TP + FN)
            fpr[i,j] = FP/(FP + TN)
            
    return names, time_db, time_identification, accuracy, precision, recall, fpr

