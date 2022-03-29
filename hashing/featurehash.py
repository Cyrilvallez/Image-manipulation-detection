#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:48:23 2022

@author: cyrilvallez
"""

from hashing.general_hash import Algorithm
from hashing.imagehash import ImageHash, ImageMultiHash
import skimage.feature as feature


def orb(image):
    extractor = feature.ORB(n_keypoints=5)
    
    img = image.convert('L')
    extractor.detect_and_extract(img)
    
    hashes = []
    for hash_ in extractor.descriptors:
        hashes.append(ImageHash(hash_))
    
    return ImageMultiHash(hashes)

# Mapping from string to actual algorithms
FEATURE_MODEL_SWITCH = {
    'ORB': orb,
    }


class FeatureAlgorithm(Algorithm):
    """
    Wrapper class to represent together a feature algorithm and its parameters
    
    Attributes
     ----------
     algorithm : str
         The name of the model.
     batch_size : int, optional
         Batch size for the database creation. The default is 512.
         
    """
    
    def __init__(self, algorithm, batch_size=512):
        
        Algorithm.__init__(self, algorithm, batch_size=batch_size)
        self.algorithm = FEATURE_MODEL_SWITCH[algorithm]
        
        
    def __str__(self):
        
        return f'{self.name}'
        
    
    def process_batch(self, preprocessed_images):
        """
        Process a batch of imgs and convert to a list of fingerprints.

        Parameters
        ----------
        preprocessed_images : List or Tuple of PIL images
            List representing a batch of images.

        Returns
        -------
        hashes : List
            The fingerprints corresponding to the batch of images.

        """
        
        hashes = []
        
        for image in preprocessed_images:
            hashes.append(self.algorithm(image))
            
        return hashes