#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:48:23 2022

@author: cyrilvallez
"""

from hashing.general_hash import Algorithm
from hashing.imagehash import ImageHash, ImageMultiHash
import cv2
import numpy as np

def array_of_bytes_to_bits(array):
    
    out = []
    
    for byte in array:
        bits = [True if digit=='1' else False for digit in f'{byte:08b}']
        out.extend(bits)
        
    return np.array(out)


class ImageDescriptors(object):
    """
    Image descriptors encapsulation. Can be used for easy comparisons with other 
    ImageDescriptors or databases of ImageDescriptors.
    """
    
    def __init__(self, descriptors, matcher):
        self.descriptors = descriptors
        self.matcher = matcher
        
    def __str__(self):
        return str(self.features)

    def __repr__(self):
        return repr(self.features)
    
    def __eq__(self, other):
        assert(self.descriptors.shape == other.descriptors.shape)
        return np.allclose(self.descriptors, other.descriptors)

    def __ne__(self, other):
        assert(self.descriptors.shape == other.descriptors.shape)
        return not np.allclose(self.descriptors, other.descriptors)

    def __len__(self):
        return len(self.descriptors)
    
    
    def matches(self, other, threshold, cutoff=1):
        """
        Check if the distance between current ImageDescriptors and another
        ImageDescriptors is less than a threshold, for at least cutoff features.

        Parameters
        ----------
        other : ImageDescriptors
            The other ImageDescriptors.
        threshold : float
            Threshold for distance identification.
        cutoff : int, optional
            The number of descriptor that must be lower than the threshold for
            a match. The default is 1.

        Returns
        -------
        Boolean
            Whether or not there is a match.

        """
        
        matches = self.matcher.match(self.descriptors, other.descriptors)
        matches = sorted(matches, key = lambda x: x.distance)
        
        return matches[cutoff-1].distance <= threshold
    
    
    def match_db(self, database, threshold, cutoff=1):
        """
        Check if there is a ImageDescriptors in the database for which the distance
        with current ImageDescriptors is less than a threshold.

        Parameters
        ----------
        database : Dictionary
            Dictionary of type {'img_name':ImageDescriptors}. Represents the database.
        threshold : float
            Threshold for distance identification.
        cutoff : int, optional
            The number of descriptor that must be lower than the threshold for
            a match. The default is 1.

        Returns
        -------
        Boolean
            Whether or not there is a match in the database.

        """
        
        for descriptors in database.values():
            if self.matches(descriptors, threshold=threshold, cutoff=cutoff):
                return True
        return False
    
    
    def match_db_image(self, database, threshold, cutoff=1):
        """
        Check if the current descriptors match other descriptors in the database.

        Parameters
        ----------
        database : Dictionary
            Dictionary of type {'img_name':ImageDescriptors}. Represents the database.
        threshold : float
            Threshold for distance identification.
        cutoff : int, optional
            The number of descriptor that must be lower than the threshold for
            a match. The default is 1.

        Returns
        -------
        names : List
            Name of all images in the database which trigger a similarity with
            current ImageDescriptors.

        """
        
        names = []
        for key in database.keys():
            if self.matches(database[key], threshold=threshold, cutoff=cutoff):
                names.append(key)
        
        return names
    
    


def ORB(image, n_features=20, device='cuda'):
    
    img = np.array(image.convert('L'))
    
    if device=='cuda':
        src = cv2.cuda_GpuMat()
        src.upload(img)
        
        orb = cv2.cuda.ORB_create(nfeatures=n_features)
        _, features = orb.detectAndComputeAsync(img, cv2.cuda_Stream.Null())
        descriptors = features.download()
        
    elif device=='cpu':
        orb = cv2.ORB_create(nfeatures=n_features)
        _, descriptors = orb.detectAndCompute(img, None)
    
    return descriptors


# Mapping from string to actual algorithms
FEATURE_MODEL_SWITCH = {
    'ORB': ORB,
    }


ALGORITHMS_MATCHER = {
    'ORB': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
    
    def __init__(self, algorithm, batch_size=512, n_features=20, device='cuda'):
        
        Algorithm.__init__(self, algorithm, batch_size=batch_size)
        
        if (device not in ['cuda', 'cpu']):
            raise ValueError('device must be either `cuda` or `cpu`.')
            
        self.algorithm = FEATURE_MODEL_SWITCH[algorithm]
        self.matcher = ALGORITHMS_MATCHER[algorithm]
        self.n_features = n_features
        self.device = device
        
    def __str__(self):
        
        return f'{self.name} {self.n_features} features'
        
    
    def process_batch(self, preprocessed_images):
        """
        Process a batch of imgs and convert to a list of fingerprints.

        Parameters
        ----------
        preprocessed_images : List or Tuple of PIL images
            List representing a batch of images.

        Returns
        -------
        fingerprints : List
            The fingerprints corresponding to the batch of images.

        """
        
        fingerprints = []
        
        for image in preprocessed_images:
            descriptors = self.algorithm(image, self.n_features, self.device)
            fingerprints.append(ImageDescriptors(descriptors, self.matcher))
            
        return fingerprints