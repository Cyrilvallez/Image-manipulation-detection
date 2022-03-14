#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:13:39 2022

@author: cyrilvallez
"""

from PIL import Image
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from Project import generator
from hashing.imagehash import ImageHash
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.models import inception_v3
from hashing.SimCLR import resnet_wider 


class ImageFeatures(object):
    """
    Image features encapsulation. Can be used for easy comparisons with other 
    ImageFeatures or databases of ImageFeatures.
    """
    
    def __init__(self, features):
        self.features = np.array(features).squeeze()
        if (len(self.features.shape) > 1):
            raise TypeError('ImageFeature array must be 1D')

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
    
    
    def cosine_distance(self, other):
        """
        Cosine distance between current ImageFeatures and another one.

        Parameters
        ----------
        other : ImageFeatures
            The other ImageFeatures.

        Raises
        ------
        TypeError
            If both ImageFeatures are not the same length.

        Returns
        -------
        Float
            The cosine distance between both ImageFeatures (between 0 and 1).

        """
        
        if len(self) != len(other):
            raise TypeError('ImageFeatures must be of the same length.')
        
        return 1 - 1/2 - 1/2*np.dot(self.features, other.features)/ \
            np.linalg.norm(self.features)/np.linalg.norm(other.features)


    def matches(self, other, threshold=0.25):
        """
        Check if the cosine distance between current ImageFeatures and another
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

        return self.cosine_distance(other) <= threshold
    
    
    def match_db(self, database, threshold=0.25):
        """
        Check if there is a ImageFeatures in the database for which the cosine distance
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


class ImageDataset(Dataset):
    """
    Class representing a dataset of images. Convenient to use in conjunction
    with PyTorch DataLoader.
    """
    
    def __init__(self, path_to_imgs, transforms, device='cuda'):
        if type(path_to_imgs) == str:
            self.img_paths = [path_to_imgs + name for name in os.listdir(path_to_imgs)]
        elif type(path_to_imgs) == list:
            self.img_paths = path_to_imgs
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        image = self.transforms(image)
        image = image.to(torch.device(self.device))
        try:
            name = self.img_paths[index].rsplit('/', 1)[1]
        except IndexError:
            name = self.img_paths[index]
        return image, name
    
    
class ImageIterableDataset(IterableDataset):
    """
    Class representing a dataset of attacks on images. Convenient to use in conjunction
    with PyTorch DataLoader.
    """
    
    def __init__(self, imgs_to_attack, img_names, transforms, device='cuda'):
        super(IterableDataset).__init__()
        self.imgs_to_attack = imgs_to_attack
        self.img_names = img_names
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.imgs_to_attack)
    
    def __iter__(self):
        
        for img, img_name in zip(self.imgs_to_attack, self.img_names):
            
            attacks = generator.perform_all_attacks(img, **generator.ATTACK_PARAMETERS)
            for key in attacks.keys():
                image = attacks[key]
                attack_name = key
                image = image.convert('RGB')
                image = self.transforms(image)
                image = image.to(torch.device(self.device))
                
                yield (image, img_name, attack_name)
                
                
def _partition_dataset(path_to_imgs, fraction, seed=23):
    """
    Partition a dataset by choosing a fraction `fraction` of images to attack.

    Parameters
    ----------
    path_to_imgs : Str or list of str
        The path to a directory containing images or a list of path to images.
    fraction : Float
        Fraction of images to attack in the directory if `path_to_imgs` is a str.
        If `path_to_imgs` is a list, it is ignored and all images in the list
        are attacked.
    seed : int, optional
        Fixed seed for coherent results. The default is 23.

    Returns
    -------
    imgs_to_attack : List of str
        List of paths towards images to attack.
    img_names : List of str
        Short names (without full path) of the images to attack.

    """
    
    if (type(path_to_imgs) == str or type(path_to_imgs) == np.str_):
        img_paths = [path_to_imgs + name for name in os.listdir(path_to_imgs)]
        
        rng = np.random.default_rng(seed=seed)
        imgs_to_attack = rng.choice(img_paths, size=round(len(img_paths)*fraction), 
                   replace=False)
    elif type(path_to_imgs) == list:
        imgs_to_attack = path_to_imgs
        
    img_names = []
    
    for img in imgs_to_attack:
        try:
            img_names.append(img.rsplit('/', 1)[1])
        except IndexError:
            img_names.append(img)
            
    return imgs_to_attack, img_names


def _create_db(model, path_to_imgs, transforms, raw_features=False, features_size=None, 
                     hash_size=8, batch_size=256, device='cuda'):
    """
    Creates a database of fingerprints (ImageFeatures or ImageHash).

    Parameters
    ----------
    model : Pytorch model
        The model used for neural hashing.
    path_to_imgs : Str or list of str
        The path to a directory containing images or a list of path to images.
    transforms : torchvision.transforms object
        The transforms to apply on each image for pre-processing.
    raw_features : Boolean, optional
        If true, the output features of the network are not hashed, and features are 
        returned.
    features_size : Int
        Size of the output of the `model` in the Euclidean space.
    hash_size : int, optional
        The square of the hash size (to be consistent with imagehash library).
        The default is 8, resulting in a hash of length 8**2=64.
    batch_size : int, optional
        The batch size for the network. The default is 256.
    device : str, optional
        Device to run the computations. Either `cpu` or `cuda`. The default is 'cuda'.

    Returns
    -------
    fingerprints : Dictionary
        Dictionary of all fingerprints (ImageFeatures or ImageHash) of the form
        {img_name: fingerprint}

    """
    
    # Creates the dataloader to easily iterate on images
    dataset = ImageDataset(path_to_imgs, transforms, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # random hyperplanes (with constant seed to always get the same for constant
    # hash size)
    if (not raw_features):
        rng = np.random.default_rng(seed=135)
        hyperplanes = 2*rng.rand(features_size, hash_size**2) - 1
    
    fingerprints = {}
    
    for imgs, img_names in dataloader:
        
        # Apply the pretrained model
        with torch.no_grad():
            features = model(imgs).cpu().numpy()
        
        if (not raw_features):
            # Computes the dot products between each image and the hyperplanes and
            # Select the bits depending on orientation 
            img_hashes = features@hyperplanes > 0
        
            for img_hash, img_name in zip(img_hashes, img_names):
                fingerprints[img_name] = ImageHash(img_hash)
                
        else:
            for feature, img_name in zip(features, img_names):
                fingerprints[img_name] = ImageFeatures(feature)
    
    return fingerprints
                
                
def _hashing(model, path_to_imgs, transforms, database, threshold, raw_features=False,
             features_size=None, attack_fraction=0.3, hash_size=8, batch_size=256,
             device='cuda'):
    """
    Hash images using model, and check for matches in the database.

    Parameters
    ----------
    model : Pytorch model
        The model used for neural hashing.
    path_to_imgs : Str or list of str
        The path to a directory containing images or a list of path to images.
        Attacks are performed on only `attack_fraction` if `path_to_imgs` is a
        str (directory).
    transforms : torchvision.transforms object
        The transforms to apply on each image for pre-processing.
    database : Dictionary
        The database in which to check for matches.
    threshold : Float
        Threshold for identification. 
    raw_features : Boolean, optional
        If true, the output features of the network are not hashed, and features are 
        returned.
    features_size : Int
        Size of the output of the `model` in the Euclidean space.
    attack_fraction : Float, optional
        Fraction of the directory `path_to_imgs` used to generate attacks, if 
        `path_to_imgs` is a str. If `path_to_imgs` is a list, it is ignored.
        The default is 0.3.
    hash_size : int, optional
        The square of the hash size (to be consistent with imagehash library).
        The default is 8, resulting in a hash of length 8**2=64.
    batch_size : int, optional
        The batch size for the network. The default is 256.
    device : str, optional
        Device to run the computations. Either `cpu` or `cuda`. The default is 'cuda'.
        

    Returns
    -------
    general_output : Dictionary
        Results at the global level (number of matches/misses).
    attack_wise_output : Dictionary
        Results at the attack level (number of matches/misses).
    image_wise_output : Dictionary
        Results at the image level, for images in the database (number of 
        correct/incorrect identification).

    """
    
    imgs_to_attack, imgs_to_attack_names = _partition_dataset(path_to_imgs, attack_fraction)
    
    # Creates the dataloader to easily iterate on images
    dataset = ImageIterableDataset(imgs_to_attack, imgs_to_attack_names,
                                   transforms, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # random hyperplanes (with constant seed to always get the same for constant
    # hash size)
    if (not raw_features):
        rng = np.random.default_rng(seed=135)
        hyperplanes = 2*rng.rand(features_size, hash_size**2) - 1
        
    # Initialize the different output digests
    
    # General outputs
    general_output = {'detection':0, 'no detection':0}
    
    # Attack-wise detection rate
    all_attack_names = generator.retrieve_ids(**generator.ATTACK_PARAMETERS)
    attack_wise_output = {}
    for name in all_attack_names:
        attack_wise_output[name] = {'detection':0, 'no detection':0}
        
    # Image-wise detection rate
    image_wise_output = {}
    for name in database.keys():
        image_wise_output[name] = {'correct detection':0, 'uncorrect detection':0}
    
    
    for imgs, img_names, attack_names in dataloader:
        # Apply the pretrained model
        with torch.no_grad():
            features = model(imgs).cpu().numpy()
        
        if (not raw_features):
            # Computes the dot products between each image and the hyperplanes and
            # Select the bits depending on orientation 
            img_hashes = features@hyperplanes > 0
        
            for img_hash, img_name, attack_name in zip(img_hashes, img_names, attack_names):
                img_hash = ImageHash(img_hash)
                detected = img_hash.match_db_image(database, threshold=threshold)
                
                if len(detected) > 0:
                    general_output['detection'] += 1
                    attack_wise_output[attack_name]['detection'] += 1
                else:
                    general_output['no detection'] += 1
                    attack_wise_output[attack_name]['no detection'] += 1
                
                for name in detected:
                    if name == img_name:
                        image_wise_output[name]['correct detection'] += 1
                    else:
                        image_wise_output[name]['uncorrect detection'] += 1
                
        else:
            
            for feature, img_name, attack_name in zip(features, img_names, attack_names):
                feature = ImageFeatures(feature)
                detected = feature.match_db_image(database, threshold=threshold)
                
                if len(detected) > 0:
                    general_output['detection'] += 1
                    attack_wise_output[attack_name]['detection'] += 1
                else:
                    general_output['no detection'] += 1
                    attack_wise_output[attack_name]['no detection'] += 1
                
                for name in detected:
                    if name == img_name:
                        image_wise_output[name]['correct detection'] += 1
                    else:
                        image_wise_output[name]['uncorrect detection'] += 1
    
    return (general_output, attack_wise_output, image_wise_output)



def load_inception(device='cuda'):
    
    assert (device=='cpu' or device=='cuda')
    
    # Load the model 
    inception = inception_v3(pretrained=True, transform_input=False)
    # Overrides last Linear layer
    inception.fc = nn.Identity()
    inception.eval()
    inception.to(torch.device(device))
        
    # Process the image 
    transforms = T.Compose([
        T.Resize((299,299), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return inception, transforms



def load_simclr(device='cuda'):
    
    
    assert (device=='cpu' or device=='cuda')
    
    # Load the model 
    simclr = resnet_wider.resnet50x2()
    try:
        checkpoint = torch.load(os.path.expanduser('~/Project/hashing/SimCLR/resnet50-2x.pth'))
    except FileNotFoundError:
        checkpoint = torch.load(os.path.expanduser('~/Desktop/Project/hashing/SimCLR/resnet50-2x.pth'))
    simclr.load_state_dict(checkpoint['state_dict'])
    simclr.fc = nn.Identity()
    simclr.eval()
    simclr.to(torch.device(device))

    # No normalization as the original model
    transforms = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(224),
        T.ToTensor()
        ])
    
    return simclr, transforms


# Mapping from string to actual algorithms
NEURAL_MODEL_LOADER = {
    'Inception': load_inception,
    'SimCLR': load_simclr
    }


# Mapping from string to feature size output of networks
NEURAL_MODEL_FEATURES_SIZE = {
    'Inception': 2048,
    'SimCLR': 4096
    }


class NeuralAlgorithm(object):
    """
    Wrapper class to represent together an algorithm and its parameters (hash_size,
    batch_size, device,...)
    """
    
    def __init__(self, model, hash_size=8, batch_size=256, device='cuda',
                 raw_features=False):
        if (model not in NEURAL_MODEL_LOADER.keys()):
            raise ValueError(f'model must be one of {list(NEURAL_MODEL_LOADER.keys())}')
            
        if (device not in ['cuda', 'cpu']):
            raise ValueError('device must be either `cuda` or `cpu`.')
            
        self.loader = NEURAL_MODEL_LOADER[model]
        self.name = model
        self.hash_size = hash_size
        self.features_size = NEURAL_MODEL_FEATURES_SIZE[model]
        self.raw_features = raw_features
        self.batch_size = batch_size
        self.device = device
        
    def __str__(self):
        if self.raw_features:
            return f'{self.name} raw features'
        else:
            return f'{self.name} {self.hash_size**2} bits'
        
    def __call__(self, path_to_imgs, threshold, attack_fraction=0.3, database=None):
        
        return self.algorithm(path_to_imgs, database=database, threshold=threshold, raw_features=self.raw_features,
               features_size=self.features_size, hash_size=self.hash_size,
               attack_fraction=attack_fraction, batch_size=self.hash_size,
               device=self.device)


    def create_database(self, path_to_imgs):
        
        model, transforms = self.loader(self.device)
        
        return _create_db(model, path_to_imgs, transforms, raw_features=self.raw_features,
                          features_size=self.features_size, hash_size=self.hash_size,
                          batch_size=self.batch_size, device=self.device)
    
    
    def match_database(self, path_to_imgs, database, threshold, attack_fraction=0.3):
        
        model, transforms = self.loader(self.device)
        
        return _hashing(path_to_imgs, database=database, threshold=threshold,
                        raw_features=self.raw_features, features_size=self.features_size,
                        hash_size=self.hash_size, attack_fraction=attack_fraction,
                        batch_size=self.hash_size, device=self.device)
