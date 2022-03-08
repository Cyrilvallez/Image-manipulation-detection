#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:13:39 2022

@author: cyrilvallez
"""

from PIL import Image
import numpy as np
import os
from hashing.imagehash import ImageHash
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3
from hashing.SimCLR import resnet_wider 


class ImageDataset(Dataset):
    """
    Class representing a dataset of images. Convenient to use in conjunction
    with PyTorch DataLoader.
    """
    
    def __init__(self, path_to_imgs, transforms, device='cuda'):
        self.img_paths = [path_to_imgs + name for name in os.listdir(path_to_imgs)]
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        image = self.transforms(image)
        image = image.to(torch.device(self.device))
        return image
    

# Initialize the model here for future global use
INCEPTION = None

def inception_hash(path_to_imgs, hash_size=8, batch_size=256, device='cuda'):
    """
    Neural hash using the pretrained inception v3 pytorch model (pretrained on
    ImageNet)

    Parameters
    ----------
    path_to_imgs : str
        Path to a folder containing the images to hash.
    hash_size : int, optional
        The square of the hash size (to be consistent with imagehash library).
        The default is 8, resulting in a hash of length 8**2=64.
    batch_size : int, optional
        The batch size for the network. The default is 256.
    device : str, optional
        Device to run the computations. Either `cpu` or `cuda`. The default is 'cuda'.

    Returns
    -------
    hashes : list of hashes
        Hashes for all the images in the folder `path_to_imgs`.

    """
    
    assert (device=='cpu' or device=='cuda')
    
    # The model is defined as global so that it is not loaded at every call
    global INCEPTION
    # Load the model if it has not being loaded already
    if INCEPTION is None:
        INCEPTION = inception_v3(pretrained=True, transform_input=False)
        # Overrides last Linear layer
        INCEPTION.fc = nn.Identity()
        INCEPTION.eval()
        INCEPTION.to(torch.device(device))
        
    # Process the image 
    transforms = T.Compose([
        T.Resize((299,299), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Creates the dataloader to easily iterate on images
    dataset = ImageDataset(path_to_imgs, transforms, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # random hyperplanes (with constant seed to always get the same)
    np.random.seed(135)
    hyperplanes = 2*np.random.rand(2048, hash_size**2) - 1
    
    hashes = []
    
    for imgs in dataloader:
        # Apply the pretrained model
        with torch.no_grad():
            features = INCEPTION(imgs).cpu().numpy()
        # Computes the dot products between each image and the hyperplanes and
        # Select the bits depending on orientation 
        img_hashes = features@hyperplanes > 0
        
        for img_hash in img_hashes:
            hashes.append(ImageHash(img_hash))
        
    
    return hashes



# Initialize the model here for future global use
SIMCLR = None

def simclr_hash(path_to_imgs, hash_size=8, batch_size=256, device='cuda'):
    """
    Neural hash using the pretrained SimCLR model with resnet50-2x as architecture
    (pretrained on ImageNet, and ported from tensorflow)

    Parameters
    ----------
    path_to_imgs : str
        Path to a folder containing the images to hash.
    hash_size : int, optional
        The square of the hash size (to be consistent with imagehash library).
        The default is 8, resulting in a hash of length 8**2=64.
    batch_size : int, optional
        The batch size for the network. The default is 256.
    device : str, optional
        Device to run the computations. Either `cpu` or `cuda`. The default is 'cuda'.

    Returns
    -------
    hashes : list of hashes
        Hashes for all the images in the folder `path_to_imgs`.

    """
    
    # The model is defined as global so that it is not loaded at every call
    global SIMCLR
    # Load the model if it has not being loaded already
    if SIMCLR is None:
        SIMCLR = resnet_wider.resnet50x2()
        CHECKPOINT = torch.load(os.path.dirname(os.getcwd()) + '/hashing/SimCLR/resnet50-2x.pth')
        SIMCLR.load_state_dict(CHECKPOINT['state_dict'])
        SIMCLR.fc = nn.Identity()
        SIMCLR.eval()
        SIMCLR.to(torch.device(device))

    # No normalization as the original model
    transforms = T.Compose([
        T.Resize(256, interpolation=Image.LANCZOS),
        T.CenterCrop(224),
        T.ToTensor()
        ])
    
    # Creates the dataloader to easily iterate on images
    dataset = ImageDataset(path_to_imgs, transforms, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # random hyperplanes (with constant seed to always get the same)
    np.random.seed(135)
    hyperplanes = 2*np.random.rand(4096, hash_size**2) - 1
    
    hashes = []

    for imgs in dataloader:
        # Apply the pretrained model
        with torch.no_grad():
            features = SIMCLR(imgs).cpu().numpy()
        # Computes the dot products between each image and the hyperplanes and
        # Select the bits depending on orientation 
        img_hashes = features@hyperplanes > 0
        
        for img_hash in img_hashes:
            hashes.append(ImageHash(img_hash))
            
    
    return hashes



# Initialize the model here for future global use
SIMCLR_FEATURES = None

def simclr_features(path_to_imgs, hash_size=8, batch_size=256, device='cuda'):
    
    # The model is defined as global so that it is not loaded at every call
    global SIMCLR_FEATURES
    # Load the model if it has not being loaded already
    if SIMCLR is None:
        SIMCLR_FEATURES = resnet_wider.resnet50x2()
        CHECKPOINT = torch.load(os.path.dirname(os.getcwd()) + '/hashing/SimCLR/resnet50-2x.pth')
        SIMCLR_FEATURES.load_state_dict(CHECKPOINT['state_dict'])
        SIMCLR_FEATURES.fc = nn.Identity()
        SIMCLR_FEATURES.eval()
        SIMCLR_FEATURES.to(torch.device(device))

    # No normalization as the original model
    transforms = T.Compose([
        T.Resize(256, interpolation=Image.LANCZOS),
        T.CenterCrop(224),
        T.ToTensor()
        ])
    
    # Creates the dataloader to easily iterate on images
    dataset = ImageDataset(path_to_imgs, transforms, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []

    for imgs in dataloader:
        # Apply the pretrained model
        with torch.no_grad():
            feature_imgs = SIMCLR(imgs).cpu().numpy()
            
        for feature in feature_imgs:
            features.append(feature)
        

    return features

def cosine_similarity(a, b):
    """
    cosine similarity between 0 and 1
    """
    assert(len(a) == len(b))
    
    return 1/2 + 1/2*np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)

def features_matching(a, b, threshold):
    
    return cosine_similarity(a, b) <= threshold

def match_db(a, db, threshold):
    
    for feature in db:
        if features_matching(a, feature, threshold):
            return True
    return False
