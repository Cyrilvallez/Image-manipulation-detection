#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:14:08 2022

@author: cyrilvallez
"""

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from Project import generator
from hashing.imagehash import imagehash as ih
from hashing.imagehash import neuralhash as nh
import time
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
from PIL import Image

class ImageDataset(Dataset):
    """
    Class representing a dataset of images. Convenient to use in conjunction
    with PyTorch DataLoader.
    """
    
    def __init__(self, path_to_imgs):
        if (type(path_to_imgs) == str or type(path_to_imgs) == np.str_):
            self.img_paths = [path_to_imgs + name for name in os.listdir(path_to_imgs)]
        elif type(path_to_imgs) == list:
            self.img_paths = path_to_imgs

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        try:
            name = self.img_paths[index].rsplit('/', 1)[1]
        except IndexError:
            name = self.img_paths[index]
            
        return (image, name)
                
                
                
class ImageIterableDataset(IterableDataset):
    """
    Class representing a dataset of attacks on images. Convenient to use in conjunction
    with PyTorch DataLoader.
    """
    
    def __init__(self, imgs_to_attack):
        super(IterableDataset).__init__()
        self.imgs_to_attack = imgs_to_attack

    def __len__(self):
        return len(self.imgs_to_attack)
    
    def __iter__(self):
        
        for image in self.imgs_to_attack:
            
            try:
                img_name = image.rsplit('/', 1)[1]
            except IndexError:
                img_name = image
            
            attacks = generator.perform_all_attacks(image, **generator.ATTACK_PARAMETERS)
            for key in attacks.keys():
                image = attacks[key]
                attack_name = key
                image = image.convert('RGB')
                
                yield (image, img_name, attack_name)
                

def collate(batch):
    """
    Custom collate function to use with PyTorch dataloader

    Parameters
    ----------
    batch : List of tuples 
        Corresponds to a batch of examples from a dataset.

    Returns
    -------
    Tuple
        Tuple representing a full batch containing a tuple of PIL images and
        other tuples of names.

    """
    if (len(batch[0]) == 2):
        imgs, names = zip(*batch)
        return (imgs, names)
    
    elif (len(batch[0]) == 3):
        imgs, names, attacks = zip(*batch)
        return (imgs, names, attacks)
                
                

ADMISSIBLE_ALGORITHMS = list(nh.NEURAL_MODEL_LOADER.keys()) + \
    list(ih.CLASSICAL_MODEL_SWITCH.keys())
                
                
class Algorithm(object):
    """
    Wrapper parent class to represent together an algorithm and its parameters (algorithm,
    hash_size...)
    """
    
    def __init__(self, algorithm, hash_size=8, batch_size=512):
        
        if (algorithm not in ADMISSIBLE_ALGORITHMS):
            raise ValueError(f'model must be one of {ADMISSIBLE_ALGORITHMS}')
            
        self.name = algorithm
        self.hash_size = hash_size
        self.batch_size = batch_size
        
    def __str__(self):
            return f'{self.name} {self.hash_size**2} bits'


    def create_database(self, path_to_db):
        
        self.load_model()
        
        # Creates the dataloader to easily iterate on images
        dataset = ImageDataset(path_to_db)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                collate_fn=collate)
        
        database = {}
        
        for images, image_names in dataloader:
            
            imgs = self.preprocess(images)
            fingerprints = self.process_batch(imgs)
            
            for i, name in enumerate(image_names):
                database[name] = fingerprints[i]
        
        self.kill_model()
        
        return database
    
    
    def preprocess(self, img_list):
        """
        Preprocess a list of PIL images for use in the different algorithms. 
        By default, does nothing as needed for child ClassicalAlgorithm.

        Parameters
        ----------
        img_list : List
            List of PIL images.

        Returns
        -------
        img_list : List
            The pre-processed images.

        """
        return img_list
    
    
    def load_model(self):
        """
        Does nothing by default. Will be overriden in child NeuralAlgorithm
        for loading the model onto GPU (or CPU).
        """
        pass
        
    def kill_model(self):
        """
        Does nothing by default. Will be overriden in child NeuralAlgorithm
        for deleting the model from GPU (or CPU).
        """
        pass
        
    def process_batch(self, preprocessed_images):
        """
        Virtual method. Will be overriden in all child classes to process a batch of images and
        return a list of ImageHash or ImageFeatures objects.
        """
        raise NotImplementedError()
        
    
    
def partition_dataset(path_to_imgs, fraction, seed=23):
    """
    Randomly chooses a fraction `fraction` of images to attack in a dataset.

    Parameters
    ----------
    path_to_imgs : Str or list of str
        The path to a directory containing images or a list of path to images.
        See fraction for details.
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

    """
    
    if (type(path_to_imgs) == str or type(path_to_imgs) == np.str_):
        img_paths = [path_to_imgs + name for name in os.listdir(path_to_imgs)]
        
    elif type(path_to_imgs) == list:
        img_paths = path_to_imgs
        
    rng = np.random.default_rng(seed=seed)
    imgs_to_attack = rng.choice(img_paths, size=round(len(img_paths)*fraction), 
               replace=False)
            
    return IterableDataset(imgs_to_attack)
  
                
                
def hashing(algorithms, thresholds, databases, dataset, general_batch_size):
    """
    Performs the hashing and matching process for different algorithms and
    thresholds.
    
    Parameters
    ----------
    algorithms : List
        List of Algorithms (NeuralAlgorithm or ClassicalAlgorithm).
    thresholds : List
        List of floats corresponding to different thresholds.
    databases : List
        List of list of ImageHash or ImageFeatures, corresponding to the
        databases for each algorithm in `algorithms`.
    dataset : IterableDataset
        IterableDataset object with images to attack.
    general_batch_size : int
        Batch size for the outer Dataloader, which all algorithms will use.

    Returns
    -------
    general_output : Dictionary
        Results at the global level (number of matches/misses) for each algorith
        and threshold.
    attack_wise_output : Dictionary
        Results at the attack level (number of matches/misses) for each algorith
        and threshold.
    image_wise_output : Dictionary
        Results at the image level, for images in the database (number of 
        correct/incorrect identification) for each algorith and threshold.
    running_time : Dictionary
        Total running time (creating fingerprints and mean matching time over thresholds)
        for each algorithm.

    """
    
    # Creates dataloader on which to iterate
    dataloader = DataLoader(dataset, batch_size=general_batch_size, shuffle=False,
                            collate_fn=collate)
        
    # Initialize the different output digests; they are dictionaries with
    # meaningful names
    
    general_output = {}
    attack_wise_output = {}
    image_wise_output = {}
    running_time = {}
    
    all_attack_names = generator.retrieve_ids(**generator.ATTACK_PARAMETERS)
    
    for algorithm in algorithms:
        
        general_output[str(algorithm)] = {}
        attack_wise_output[str(algorithm)] = {}
        image_wise_output[str(algorithm)] = {}
        running_time [str(algorithm)] = 0
        
        for threshold in thresholds:
            
            general_output[str(algorithm)][f'Threshold {threshold:.3f}'] = \
                {'detection':0, 'no detection':0}
            
            attack_wise_output[str(algorithm)][f'Threshold {threshold:.3f}'] = {}
            for name in all_attack_names:
                attack_wise_output[str(algorithm)][f'Threshold {threshold:.3f}'][name] = \
                    {'detection':0, 'no detection':0}
                    
            image_wise_output[str(algorithm)][f'Threshold {threshold:.3f}'] = {}
            for name in databases[0].keys():
                image_wise_output[str(algorithm)][f'Threshold {threshold:.3f}'][name] = \
                    {'correct detection':0, 'incorrect detection':0}
            
            
    # Matching logic : First create the images with modifications, then loop
    # over each algorithm and thresholds. The creation of attacks on images is
    # the biggest overhead and thus should be the outer loop. The second 
    # important overhead is the loading of big networks, but with large
    # batches, we actually don't load them that much
    
    for images, image_names, attack_names in dataloader:
        
        for i, algorithm in enumerate(algorithms):
            
            # Load the model in memory
            algorithm.load_model()
            
            # Select database corresponding to that algorithm
            database = databases[i]
            
            t0 = time.time()
            
            # Pre-process the images
            imgs = algorithm.preprocess(images)
            # Computes the hashes or features
            fingerprints = algorithm.process_batch(imgs)
            # Take corresponding database for matching
            
            # Time needed to create the fingerprints
            running_time[str(algorithm)] += time.time() - t0
            
            t0 = time.time()
            
            for threshold in thresholds:
        
                for fingerprint, img_name, attack_name in zip(fingerprints, image_names,
                                                              attack_names):
                 
                    detected = fingerprint.match_db_image(database, threshold=threshold)
                
                    if len(detected) > 0:
                        general_output[str(algorithm)][f'Threshold {threshold:.3f}'] \
                            ['detection'] += 1
                        attack_wise_output[str(algorithm)][f'Threshold {threshold:.3f}'] \
                            [attack_name]['detection'] += 1
                    else:
                        general_output[str(algorithm)][f'Threshold {threshold:.3f}'] \
                            ['no detection'] += 1
                        attack_wise_output[str(algorithm)][f'Threshold {threshold:.3f}'] \
                            [attack_name]['no detection'] += 1
                
                    for name in detected:
                        if name == img_name:
                            image_wise_output[str(algorithm)][f'Threshold {threshold:.3f}'] \
                                [name]['correct detection'] += 1
                        else:
                            image_wise_output[str(algorithm)][f'Threshold {threshold:.3f}'] \
                                [name]['incorrect detection'] += 1
                                
                            
            # Mean time needed for the matching of batch of hashes
            running_time[str(algorithm)] += (time.time() - t0)/len(thresholds)
                
        # Removes the model from memory
        algorithm.kill_model()
    
    
    return (general_output, attack_wise_output, image_wise_output, running_time)


