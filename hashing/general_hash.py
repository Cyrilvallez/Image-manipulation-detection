#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:14:08 2022

@author: cyrilvallez
"""

# =============================================================================
# Contains the hashing pipeline, linking neural and classical hashing in a
# single fremework.
# =============================================================================

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import generator
import time
from torch.utils.data import Dataset, IterableDataset, DataLoader
from helpers import utils
import numpy as np
from PIL import Image
from tqdm import tqdm

class DatabaseDataset(Dataset):
    """
    Class representing a dataset of images to create the database. Convenient to 
    use in conjunction with PyTorch DataLoader.
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
            
        # Removes the extension (name.jpg -> name)
        name = name.rsplit('.', 1)[0]
            
        return (image, name)
    
    

class ExistingAttacksDataset(Dataset):
    """
    Class representing a dataset of existing attacked images. Convenient to use in conjunction
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
            
        # Assumes that the filename convention is name_attackID.extension
        name, attack_name = name.split('_', 1)
        # removes the extension
        attack_name = attack_name.rsplit('.', 1)[0]
            
        return (image, name, attack_name)
                
                
                
class PerformAttacksDataset(IterableDataset):
    """
    Class representing a dataset where we perform attacks on images on the fly. 
    Convenient to use in conjunction with PyTorch DataLoader.
    """
    
    def __init__(self, imgs_to_attack):
        super(IterableDataset).__init__()
        self.imgs_to_attack = imgs_to_attack

    def __len__(self):
        return len(self.imgs_to_attack)*generator.NUMBER_OF_ATTACKS
    
    def __iter__(self):
        
        for image in self.imgs_to_attack:
            
            try:
                img_name = image.rsplit('/', 1)[1]
            except IndexError:
                img_name = image
                
            # removes the extension
            img_name = img_name.rsplit('.', 1)[0]
            
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
                
                

ADMISSIBLE_ALGORITHMS = [
    'Ahash',
    'Phash',
    'Dhash',
    'Whash',
    'Crop resistant hash',
    'ORB',
    'SIFT',
    'FAST + DAISY',
    'FAST + LATCH',
    'Inception v3',
    'ResNet50 1x',
    'ResNet101 1x',
    'ResNet152 1x',
    'ResNet50 2x',
    'ResNet101 2x',
    'EfficientNet B7',
    'SimCLR v1 ResNet50 1x',
    'SimCLR v1 ResNet50 2x',
    'SimCLR v1 ResNet50 4x',
    'SimCLR v2 ResNet50 2x',
    'SimCLR v2 ResNet101 2x',
    'SimCLR v2 ResNet152 3x',
    ]
                
                
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


    def create_database(self, path_to_db, time_database={}):
        
        self.load_model()
        
        # Creates the dataloader to easily iterate on images
        dataset = DatabaseDataset(path_to_db)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                collate_fn=collate)
        
        t0 = time.time()
        
        database = {}
        
        for images, image_names in dataloader:
            
            imgs = self.preprocess(images)
            fingerprints = self.process_batch(imgs)
            
            for i, name in enumerate(image_names):
                database[name] = fingerprints[i]
                
        time_database[str(self)] = time.time() - t0
        
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
        
        
def find_attacked_images(dataset):
    """
    Find the names of images which are going to be attacked, or are already attacked

    Parameters
    ----------
    dataset : PerformAttacksDataset or ExistingAttacksDataset
        The dataset corresponding to the images.

    Returns
    -------
    names : List
        The names of the images.

    """
    # if dataset is a PerformAttacksDataset dataset
    try:
        images = dataset.imgs_to_attack
        try:
            names = [im.rsplit('/', 1)[1].rsplit('.', 1)[0] for im in images]
        except IndexError:
            names = [im.rsplit('.', 1)[0] for im in images]
        
    # dataset is a ExistingAttacksDataset dataset
    except AttributeError:
        images = dataset.img_paths
        try:
            names = [im.rsplit('/', 1)[1].split('_', 1)[0] for im in images]
        except IndexError:
            names = [im.split('_', 1)[0] for im in images]
        # Removes duplicates
        names = list(set(names))
        
    return names
        
    
    
def create_dataset(path_to_imgs, fraction=0.3, existing_attacks=False, seed=23):
    """
    Randomly chooses a fraction `fraction` of images to attack in a dataset.

    Parameters
    ----------
    path_to_imgs : Str or list of str
        The path to a directory containing images or a list of path to images.
        See fraction for details.
    fraction : Float, optional
        Fraction of images to attack in the directory if `path_to_imgs` is a str.
        If `path_to_imgs` is a list, it is ignored and all images in the list
        are attacked. The default is 0.3.
    existing_attacks : Boolean, optional
        Whether the attacks are already on disk or if they need to be created
        on the fly. The default is False.
    seed : int, optional
        Fixed seed for coherent results. The default is 23.

    Returns
    -------
    ExistingAttacksDataset or PerformAttacksDataset
        Dataset of images to attack.

    """
    
    if existing_attacks:
        return ExistingAttacksDataset(path_to_imgs)
    
    else:
    
        if (type(path_to_imgs) == str or type(path_to_imgs) == np.str_):
            img_paths = [path_to_imgs + name for name in os.listdir(path_to_imgs)]
        
        elif type(path_to_imgs) == list:
            img_paths = path_to_imgs
        
        rng = np.random.default_rng(seed=seed)
        imgs_to_attack = rng.choice(img_paths, size=round(len(img_paths)*fraction), 
                                    replace=False)
            
        return PerformAttacksDataset(imgs_to_attack)
    
    
def create_databases(algorithms, path_to_db):
    """
    Creates the databases for each algorithm.

    Parameters
    ----------
    algorithms : List
        List of Algorithms (NeuralAlgorithm or ClassicalAlgorithm).
    path_to_db : str or list of str
        The path to the folder containing the images for the database (or list
        of paths corresponding to the database).

    Returns
    -------
    databases : List
        The databases for each algorithm.
    time_database : Dictionary
        The time needed to create the database for each algorithm.

    """
    
    databases = []
    time_database = {}

    for algorithm in tqdm(algorithms):
        databases.append(algorithm.create_database(path_to_db, time_database))
        
    return databases, time_database


def get_distances(fingerprints, database):
    
    distances = []
    for fingerprint in fingerprints:
        distances.append(fingerprint.compute_distances(database))
        
    return distances


def is_detected(distances, threshold):
    
    distances_, names = distances
            
    return names[distances_ <= threshold]


def hashing(algorithms, thresholds, databases, dataset, general_batch_size=512):
    """
    Performs the hashing and matching process for different algorithms and
    thresholds.
    
    Parameters
    ----------
    algorithms : List
        List of Algorithms (NeuralAlgorithm or ClassicalAlgorithm).
    thresholds : Array 
        List of floats corresponding to different thresholds. Or array of arrays
        of the same size as algorithms, corresponding to different thresholds
        for each algorithm
    databases : List
        List of dictionaries of ImageHash or ImageFeatures, corresponding to the
        databases for each algorithm in `algorithms`.
    dataset : IterableDataset
        IterableDataset object with images to attack.
    general_batch_size : int, optional
        Batch size for the outer Dataloader, which all algorithms will use. The
        default is 512.

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
    
    # Check whether the thresholds are different for each algo or not
    if (type(thresholds[0]) == list or type(thresholds[0]) == np.ndarray):
        assert(len(thresholds) == len(algorithms))
        custom_thresholds = True
    else:
        custom_thresholds = False
        
    # Initialize the different output digests; they are dictionaries with
    # meaningful names
    
    general_output = {}
    attack_wise_output = {}
    image_wise_output = {}
    running_time = {}
    
    all_attack_names = generator.retrieve_ids(**generator.ATTACK_PARAMETERS)
    
    for i, algorithm in enumerate(algorithms):
        
        general_output[str(algorithm)] = {}
        attack_wise_output[str(algorithm)] = {}
        image_wise_output[str(algorithm)] = {}
        running_time [str(algorithm)] = 0
        
        if custom_thresholds:
            algo_thresholds = thresholds[i]
        else:
            algo_thresholds = thresholds
        
        for threshold in algo_thresholds:
            
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
    
    for images, image_names, attack_names in tqdm(dataloader):
        
        for i, algorithm in enumerate(algorithms):
            
            # Load the model in memory
            algorithm.load_model()
            
            # Select database corresponding to that algorithm
            database = databases[i]
            
            # Select thresholds corresponding to that algorithm
            if custom_thresholds:
                algo_thresholds = thresholds[i]
            else:
                algo_thresholds = thresholds
            
            t0 = time.time()
            
            # Pre-process the images
            imgs = algorithm.preprocess(images)
            # Computes the hashes or features
            fingerprints = algorithm.process_batch(imgs)
            # Get distances for all fingerprints
            distances = get_distances(fingerprints, database)
            
            # Time needed to create the fingerprints
            running_time[str(algorithm)] += time.time() - t0
            
            t0 = time.time()
            
            for threshold in algo_thresholds:
        
                for distances_, img_name, attack_name in zip(distances, image_names,
                                                              attack_names):
                 
                    detected = is_detected(distances_, threshold)
                
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
            running_time[str(algorithm)] += (time.time() - t0)/len(algo_thresholds)
                
            # Removes the model from memory
            algorithm.kill_model()
    
    
    return (general_output, attack_wise_output, image_wise_output, running_time)

                
                




def total_hashing(algorithms, thresholds, path_to_db, positive_dataset,
                  negative_dataset, general_batch_size=512):
    """
    Perform the hashing and matchup of both a experimental and control group
    of images, and outputs the (processed) metrics of the experiment.

    Parameters
    ----------
    algorithms : List
        List of Algorithms (NeuralAlgorithm or ClassicalAlgorithm).
    thresholds : Array 
        List of floats corresponding to different thresholds. Or array of arrays
        of the same size as algorithms, corresponding to different thresholds
        for each algorithm
    path_to_db : str or list of str
        The path to the folder containing the images for the database (or list
        of paths corresponding to the database).
    positive_dataset : ExistingAttacksDataset or PerformAttacksDataset
        Dataset with attacked versions of images in the database.
    negative_dataset : ExistingAttacksDataset or PerformAttacksDataset
        Dataset with attacked versions of images not present in the database.
    general_batch_size : int, optional
        Batch size for the outer Dataloader, which all algorithms will use. The
        default is 512.

    Returns
    -------
    final_digest : Tuple of dictionaries
        The metrics of the experiment for each algorithm and thresholds.

    """
    
    databases, time_database = create_databases(algorithms, path_to_db)
    
    positive_digest = hashing(algorithms, thresholds, databases, positive_dataset,
                              general_batch_size)
    negative_digest = hashing(algorithms, thresholds, databases, negative_dataset,
                              general_batch_size)
    
    attacked_image_names = find_attacked_images(positive_dataset)
    
    final_digest = utils.process_digests(positive_digest, negative_digest,
                                         attacked_image_names)
    final_digest = (*final_digest, time_database)
    
    return final_digest





























def hashing_(algorithms, thresholds, databases, dataset, general_batch_size=512):
    """
    Performs the hashing and matching process for different algorithms and
    thresholds.
    
    Parameters
    ----------
    algorithms : List
        List of Algorithms (NeuralAlgorithm or ClassicalAlgorithm).
    thresholds : Array 
        List of floats corresponding to different thresholds. Or array of arrays
        of the same size as algorithms, corresponding to different thresholds
        for each algorithm
    databases : List
        List of dictionaries of ImageHash or ImageFeatures, corresponding to the
        databases for each algorithm in `algorithms`.
    dataset : IterableDataset
        IterableDataset object with images to attack.
    general_batch_size : int, optional
        Batch size for the outer Dataloader, which all algorithms will use. The
        default is 512.

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
    
    # Check whether the thresholds are different for each algo or not
    if (type(thresholds[0]) == list or type(thresholds[0]) == np.ndarray):
        assert(len(thresholds) == len(algorithms))
        custom_thresholds = True
    else:
        custom_thresholds = False
        
    # Initialize the different output digests; they are dictionaries with
    # meaningful names
    
    general_output = {}
    attack_wise_output = {}
    image_wise_output = {}
    running_time = {}
    
    all_attack_names = generator.retrieve_ids(**generator.ATTACK_PARAMETERS)
    
    for i, algorithm in enumerate(algorithms):
        
        general_output[str(algorithm)] = {}
        attack_wise_output[str(algorithm)] = {}
        image_wise_output[str(algorithm)] = {}
        running_time [str(algorithm)] = 0
        
        if custom_thresholds:
            algo_thresholds = thresholds[i]
        else:
            algo_thresholds = thresholds
        
        for threshold in algo_thresholds:
            
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
    
    for images, image_names, attack_names in tqdm(dataloader):
        
        for i, algorithm in enumerate(algorithms):
            
            # Load the model in memory
            algorithm.load_model()
            
            # Select database corresponding to that algorithm
            database = databases[i]
            
            # Select thresholds corresponding to that algorithm
            if custom_thresholds:
                algo_thresholds = thresholds[i]
            else:
                algo_thresholds = thresholds
            
            t0 = time.time()
            
            # Pre-process the images
            imgs = algorithm.preprocess(images)
            # Computes the hashes or features
            fingerprints = algorithm.process_batch(imgs)
            
            # Time needed to create the fingerprints
            running_time[str(algorithm)] += time.time() - t0
            
            t0 = time.time()
            
            for threshold in algo_thresholds:
        
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
            running_time[str(algorithm)] += (time.time() - t0)/len(algo_thresholds)
                
            # Removes the model from memory
            algorithm.kill_model()
    
    
    return (general_output, attack_wise_output, image_wise_output, running_time)