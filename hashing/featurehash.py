#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:48:23 2022

@author: cyrilvallez
"""

# =============================================================================
# Code for keypoint-related methods
# =============================================================================

from hashing.general_hash import Algorithm
import cv2
import numpy as np

def array_of_bytes_to_bits(array):
    
    out = []
    
    for byte in array:
        bits = [True if digit=='1' else False for digit in f'{byte:08b}']
        out.extend(bits)
        
    return np.array(out)


MATCHERS = {
    'Hamming': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
    'L1': cv2.BFMatcher(cv2.NORM_L1, crossCheck=False),
    'L2': cv2.BFMatcher(cv2.NORM_L2, crossCheck=False),
    }


class ImageDescriptors(object):
    """
    Image descriptors encapsulation. Can be used for easy comparisons with other 
    ImageDescriptors or databases of ImageDescriptors.
    cutoff : int, optional
        The number of descriptor that must be lower than the threshold for
        a match. The default is 1.
    """
    
    def __init__(self, descriptors, matcher, cutoff=1):
        self.descriptors = descriptors
        self.matcher = matcher
        self.cutoff = cutoff
        
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
    
    
    def matches(self, other, threshold):
        """
        Check if the distance between current ImageDescriptors and another
        ImageDescriptors is less than a threshold, for at least cutoff features.

        Parameters
        ----------
        other : ImageDescriptors
            The other ImageDescriptors.
        threshold : float
            Threshold for distance identification.

        Returns
        -------
        Boolean
            Whether or not there is a match.

        """
        
        matches = MATCHERS[self.matcher].match(self.descriptors, other.descriptors)
        matches = sorted(matches, key = lambda x: x.distance)
        
        # Normalize the hamming distance by the number of bits in the descriptor 
        # to get the BER thresholdfe
        if (self.matcher == 'Hamming'):
            # Each value in self.descriptors[0] is a byte, thus we multiply by 8 
            # to get the total number of bits in the descriptor
            threshold *= len(self.descriptors[0])*8
        
        try:
            return matches[self.cutoff-1].distance <= threshold
        except IndexError:
            return False
    
    
    def match_db(self, database, threshold):
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
            if self.matches(descriptors, threshold=threshold):
                return True
        return False
    
    
    def match_db_image(self, database, threshold):
        """
        Check if the current descriptors match other descriptors in the database.

        Parameters
        ----------
        database : Dictionary
            Dictionary of type {'img_name':ImageDescriptors}. Represents the database.
        threshold : float
            Threshold for distance identification.

        Returns
        -------
        names : List
            Name of all images in the database which trigger a similarity with
            current ImageDescriptors.

        """
        
        names = []
        for key in database.keys():
            if self.matches(database[key], threshold=threshold):
                names.append(key)
        
        return names
    
    
    def compute_distance(self, other):
        
        if self.descriptors is None or other.descriptors is None:
            return float('inf')
        
        matches = MATCHERS[self.matcher].match(self.descriptors, other.descriptors)
        matches = sorted(matches, key = lambda x: x.distance)
        
        try:
            distance = matches[self.cutoff-1].distance 
        except IndexError:
            # Assign inf if the distance does not exist for this cutoff
            distance = float('inf')
        
        # Normalize the hamming distance by the number of bits in the descriptor 
        # to get the BER threshold
        if (self.matcher == 'Hamming'):
            # Each value in self.descriptors[0] is a byte, thus we multiply by 8 
            # to get the total number of bits in the descriptor
            distance /= len(self.descriptors[0])*8
            
        return distance
    
    
    def compute_distances(self, database):
        
        distances = []
        names = []
            
        for key in database.keys():
            distances.append(self.compute_distance(database[key]))
            names.append(key)
        
        return (np.array(distances), np.array(names))
        
        
        
        
    
    


def ORB(image, n_features=20):
    
    img = np.array(image.convert('L'))
    
    orb = cv2.ORB_create(nfeatures=n_features)
    _, descriptors = orb.detectAndCompute(img, None)
    
    return descriptors


def SIFT(image, n_features=20):
    
    img = np.array(image.convert('L'))
    
    sift = cv2.SIFT_create(nfeatures=n_features)
    _, descriptors = sift.detectAndCompute(img, None)
    
    return descriptors


def DAISY(image, n_features=20):
    
    img = np.array(image.convert('L'))
    
    detector = cv2.ORB_create(nfeatures=n_features)
    extractor = cv2.xfeatures2d.DAISY_create()
    
    kps = detector.detect(img)
    _, descriptors = extractor.compute(img, kps)

    return descriptors


def LATCH(image, n_features):
    
    img = np.array(image.convert('L'))
    
    detector = cv2.ORB_create(nfeatures=n_features)
    extractor = cv2.xfeatures2d.LATCH_create()
    
    kps = detector.detect(img)
    _, descriptors = extractor.compute(img, kps)

    return descriptors


# Mapping from string to actual algorithms
FEATURE_MODEL_SWITCH = {
    'ORB': ORB,
    'SIFT': SIFT,
    'FAST + DAISY': DAISY,
    'FAST + LATCH': LATCH,
    }


# The name of the matcher we need for each algorithms
ALGORITHMS_MATCHER = {
    'ORB': 'Hamming',
    'SIFT': 'L2',
    'FAST + DAISY': 'L2',
    'FAST + LATCH': 'Hamming',
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
    
    def __init__(self, algorithm, batch_size=512, n_features=20, matcher=None,
                 cutoff=1):
        
        try:
            Algorithm.__init__(self, algorithm, batch_size=batch_size)
        except ValueError:
            raise ValueError(f'Feature algorithm must be one of {*ALGORITHMS_MATCHER.keys(),}') 
        
        self.algorithm = FEATURE_MODEL_SWITCH[algorithm]
        if matcher is None:
            self.matcher = ALGORITHMS_MATCHER[algorithm]
        else:
            self.matcher = matcher
        self.n_features = n_features
        self.cutoff = cutoff
        
    def __str__(self):
        
        if self.cutoff == 1:
            return f'{self.name} {self.n_features} descriptors'
        else:
            return f'{self.name} {self.n_features} descriptors, cutoff {self.cutoff}'
        
    
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
            descriptors = self.algorithm(image, self.n_features)
            fingerprints.append(ImageDescriptors(descriptors, self.matcher, 
                                                 self.cutoff))
            
        return fingerprints