#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:13:39 2022

@author: cyrilvallez
"""

# =============================================================================
# Contains the neural hashing logic
# =============================================================================

import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import inception_v3
import scipy.spatial.distance as distance
from hashing.imagehash import ImageHash
from hashing.general_hash import Algorithm
from hashing.SimCLR import resnet_wider 

path = os.path.abspath(__file__)
current_folder = os.path.dirname(path)

def cosine_distance(vector, other_vector):
    """
    Cosine distance between two vectors.

    Parameters
    ----------
    vector, other_vector : arrays
        Vectors to compare.

    Raises
    ------
    TypeError
        If both vectors are not the same length.

    Returns
    -------
    Float
        The cosine distance between both vectors (between 0 and 1).

    """
    
    if len(vector) != len(other_vector):
        raise TypeError('Vectors must be of the same length.')
    
    return 1 - 1/2 - 1/2*np.dot(vector, other_vector)/ \
        np.linalg.norm(vector)/np.linalg.norm(other_vector)
        
        
def jensen_shannon_distance(vector, other_vector):
    """
    Jensen Shannon distance between two vectors.

    Parameters
    ----------
    vector, other_vector : arrays
        Vectors to compare.

    Raises
    ------
    TypeError
        If both vectors are not the same length.

    Returns
    -------
    Float
        The Jensen-Shannon distance between both vectors (between 0 and 1).

    """
   
    if len(vector) != len(other_vector):
        raise TypeError('Vectors must be of the same length.')
        
    return distance.jensenshannon(vector, other_vector, base=2)


# Distance functions to use for the distance in the case of raw features
DISTANCE_FUNCTIONS = {
'cosine': cosine_distance,
'Jensen-Shannon': jensen_shannon_distance
}


class ImageFeatures(object):
    """
    Image features encapsulation. Can be used for easy comparisons with other 
    ImageFeatures or databases of ImageFeatures.
    """
    
    def __init__(self, features, distance='Cosine'):
        self.features = np.array(features).squeeze()
        if (len(self.features.shape) > 1):
            raise TypeError('ImageFeature array must be 1D')
        self.distance_function = DISTANCE_FUNCTIONS[distance]

    def __str__(self):
        return str(self.features)

    def __repr__(self):
        return repr(self.features)

    def __eq__(self, other):
        assert(self.features.shape == other.features.shape)
        return np.allclose(self.features, other.features)

    def __ne__(self, other):
        assert(self.features.shape == other.features.shape)
        return not np.allclose(self.features, other.features)

    def __len__(self):
        return self.features.size
    

    def matches(self, other, threshold=0.25):
        """
        Check if the distance between current ImageFeatures and another
        ImageFeatures is less than a threshold.

        Parameters
        ----------
        other : ImageFeatures
            The other ImageFeatures.
        threshold : Float, optional
            Threshold for cosine distance identification. The default is 0.25.

        Returns
        -------
        Boolean
            Whether or not there is a match.

        """

        return self.distance_function(self.features, other.features) <= threshold
    
    
    def match_db(self, database, threshold=0.25):
        """
        Check if there is a ImageFeatures in the database for which the distance
        with current ImageFeatures is less than a threshold.

        Parameters
        ----------
        database : Dictionary
            Dictionary of type {'img_name':ImageFeatures}. Represents the database.
        threshold : TFloat, optional
            Threshold for cosine distance identification. The default is 0.25.

        Returns
        -------
        bool
            Whether or not there is a match in the database.

        """
        
        for feature in database.values():
            if self.matches(feature, threshold=threshold):
                return True
        return False
    
    
    def match_db_image(self, database, threshold=0.25):
        
        """
        Check if the current features match other features in the database.

        Parameters
        ----------
        database : Dictionary
            Dictionary of type {'img_name':ImageFeatures}. Represents the database.
        threshold : Float, optional
            Threshold for cosine distance identification. The default is 0.25.

        Returns
        -------
        names : List of str
            Name of all images in the database which trigger a similarity with
            current ImageFeatures.

        """
        
        names = []
        for key in database.keys():
            if self.matches(database[key], threshold=threshold):
                names.append(key)
        
        return names
    

def load_inception_v3(device='cuda'):
    """
    Load the inception net v3 from Pytorch.

    Parameters
    ----------
    device : str, optional
        Device on which to load the model. The default is 'cuda'.

    Returns
    -------
    inception : PyTorch model
        Inception net v3.

    """
    
    assert (device=='cpu' or device=='cuda')
    
    # Load the model 
    inception = inception_v3(pretrained=True, transform_input=False)
    # Overrides last Linear layer
    inception.fc = nn.Identity()
    inception.eval()
    inception.to(torch.device(device))
    
    return inception



def load_simclr_v1_resnet50_2x(device='cuda'):
    """
    Load the simclr v1 ResNet2x model.

    Parameters
    ----------
    device : str, optional
        Device on which to load the model. The default is 'cuda'.

    Returns
    -------
    simclr : PyTorch model
        The simclr v1 ResNet2x model.

    """
    
    assert (device=='cpu' or device=='cuda')
    
    # Load the model 
    simclr = resnet_wider.resnet50x2()
    checkpoint = torch.load(current_folder + '/SimCLR/resnet50-2x.pth')
    simclr.load_state_dict(checkpoint['state_dict'])
    simclr.fc = nn.Identity()
    simclr.eval()
    simclr.to(torch.device(device))
    
    return simclr


# Mapping from string to actual algorithms
NEURAL_MODEL_LOADER = {
    'Inception v3': load_inception_v3,
    'SimCLR v1 ResNet50 2x': load_simclr_v1_resnet50_2x
    }


# Mapping from string to feature size output of networks
NEURAL_MODEL_FEATURES_SIZE = {
    'Inception v3': 2048,
    'SimCLR v1 ResNet50 2x': 4096
    }

# Mapping from string to pre-processing transforms of networks
NEURAL_MODEL_TRANSFORMS = {
    'Inception v3': T.Compose([
        T.Resize((299,299), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    
    'SimCLR v1 ResNet50 2x': T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(224),
        T.ToTensor()
        ])
    }

                
class NeuralAlgorithm(Algorithm):
    """
    Wrapper class to represent together a neural algorithm and its parameters (hash_size,
    batch_size, device,...)

   Attributes
    ----------
    algorithm : str
        The name of the model.
    hash_size : Int, optional
        Square of the hash size (to be consistent with imagehash). Thus the actual
        hash size will be hash_size**2. Ignored if `raw_features` is set to True.
        The default is 8.
    raw_features : Boolean, optional
        Whether to hash the features or not. The default is False.
    distance : str, optional
        The distance function to use if `raw_features` is set to True. Ignored otherwise.
        The default is 'Cosine'.
    batch_size : int, optional
        Batch size for the database creation. The default is 512.
    device : str, optional
        The device to use for running the model. The default is 'cuda'.

    """
    
    def __init__(self, algorithm, hash_size=8, raw_features=False, distance='cosine',
                 batch_size=512, device='cuda'):
        
        Algorithm.__init__(self, algorithm, hash_size, batch_size)
            
        if (device not in ['cuda', 'cpu']):
            raise ValueError('device must be either `cuda` or `cpu`.')
            
        if (distance not in DISTANCE_FUNCTIONS.keys()):
            raise ValueError(f'Distance function must be one of {DISTANCE_FUNCTIONS.keys()}.')
            
        self.loader = NEURAL_MODEL_LOADER[algorithm]
        self.features_size = NEURAL_MODEL_FEATURES_SIZE[algorithm]
        self.transforms = NEURAL_MODEL_TRANSFORMS[algorithm]
        self.raw_features = raw_features
        self.distance = distance
        self.device = device
        
        if (not self.raw_features):
            rng = np.random.default_rng(seed=135)
            self.hyperplanes = 2*rng.random((self.features_size, self.hash_size**2)) - 1
        
    def __str__(self):
        if self.raw_features:
            return f'{self.name} raw features {self.distance}'
        else:
            return f'{self.name} {self.hash_size**2} bits'
 
    
    def preprocess(self, img_list):
        """
        Pre-process the images (from Tuple of PIL images to Tensor).

        Parameters
        ----------
        img_list : List or Tuple of PIL images
            The images to pre-process.

        Returns
        -------
        Tensor
            Pre-processed images to use in the network.

        """
        
        tensors = []
        for img in img_list:
            tensors.append(self.transforms(img))
            
        return torch.stack(tensors, dim=0).to(self.device)
    
    
    def load_model(self):
        """
        Load the model into memory.

        Returns
        -------
        None.

        """
        self.model = self.loader(self.device)
        
    def kill_model(self):
        """
        Remove the model from memory.

        Returns
        -------
        None.

        """
        del self.model
        torch.cuda.empty_cache()
        
    def process_batch(self, preprocessed_images):
        """
        Process a batch of imgs (Tensor) and convert to a list of ImageHash or
        ImageFeatures.

        Parameters
        ----------
        preprocessed_images : Tensor
            Tensor representing a batch of images.

        Raises
        ------
        AttributeError
            If the model has not been loaded.

        Returns
        -------
        fingerprints : List
            The fingerprints (ImageHash or ImageFeatures) corresponding to
            the batch of images.

        """
        
        
        with torch.no_grad():
            try:
                features = self.model(preprocessed_images).cpu().numpy()
            except AttributeError:
                raise AttributeError('The model has not been loaded before processing batch !')
                
        fingerprints = []
        
        if (not self.raw_features):
            # Computes the dot products between each image and the hyperplanes and
            # Select the bits depending on orientation 
            img_hashes = features @ self.hyperplanes > 0
            
            for img_hash in img_hashes:
                fingerprints.append(ImageHash(img_hash))
                
        else:
             
            for feature in features:
                fingerprints.append(ImageFeatures(feature, self.distance))
                
        return fingerprints  
        
        
        
        
        
        
