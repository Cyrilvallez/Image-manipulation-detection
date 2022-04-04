#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 08:29:00 2022

@author: cyrilvallez
"""

# =============================================================================
# This script generates numerous variations of an image, designed to test
# the robustness of different hashing algorithms
# =============================================================================

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from skimage import util
import cv2
import string
from io import BytesIO
from tqdm import tqdm
np.random.seed(256)

path = os.path.abspath(__file__)
current_folder = os.path.dirname(path)

def _find(path):
    """
    Internal functions intended to return a PIL image (useful to limit 
        reuse of this snippet)

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image

    Returns
    -------
    image : PIL image
        The corresponding PIL image.

    """
    
    if (type(path) == str or type(path) == np.str_):
        image = Image.open(path)
    else:
        image = path
        
    return image

# =============================================================================
# =============================================================================
# ================================ ATTACKS =====================================


def noise_attack(path, gaussian_variances=(0.01, 0.02, 0.05), 
                 speckle_variances=(0.01, 0.02, 0.05),
                 salt_pepper_amounts=(0.05, 0.1, 0.15), **kwargs):
    """
    Generates noisy versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    gaussian_variances : Tuple, optional
        Variances for the Gaussian noise. The default is (0.01, 0.02, 0.05).
    speckle_variances : Tuple, optional
        Variances for the Speckle noise. The default is (0.01, 0.02, 0.05).
    salt_pepper_amounts : Tuple, optional
        Amounts of salt and pepper. The default is (0.05, 0.1, 0.15).

    Returns
    -------
    out : Dictionary
        Noisy images variations (as PIL images)

    """
    
    image = _find(path)
    array = np.array(image)
    
    out = {}
    
    # Gaussian noise 
    for var in gaussian_variances:
        gaussian = util.random_noise(array, mode='gaussian', mean=0, var=var)
        gaussian = util.img_as_ubyte(gaussian)
        gaussian = Image.fromarray(gaussian)
        id_gaussian = 'gaussian_noise_' + str(var)
        out[id_gaussian] = gaussian
    
    # Salt and pepper noise
    for amount in salt_pepper_amounts:
        sp = util.random_noise(array, mode='s&p', amount=amount)
        sp = util.img_as_ubyte(sp)
        sp = Image.fromarray(sp)
        id_sp = 's&p_noise_' + str(amount)
        out[id_sp] = sp
      
    # Speckle noise 
    for var in speckle_variances:
        speckle = util.random_noise(array, mode='speckle', mean=0, var=var)
        speckle = util.img_as_ubyte(speckle)
        speckle = Image.fromarray(speckle)
        id_speckle = 'speckle_noise_' + str(var)
        out[id_speckle] = speckle
    
    return out


def filter_attack(path, gaussian_kernels=(3, 5, 7),
                  median_kernels=(3, 5, 7), **kwargs):
    """
    Generates filtered versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    gaussian_kernels : Tuple, optional
        Sizes for the gaussian filter. The filters will be the same size in both
        directions, and the standard deviation of the Gaussian is computed as 
        std = 0.6*((size-1)*0.5 - 1) + 1. The default is (3, 5, 7).
    median_kernels : Tuple, optional
        Sizes for the median filter. The filters will be the same size in both
        directions. The default is (3, 5, 7).

    Returns
    -------
    out : Dictionary
        Filtered variations if the image (as PIL images)

    """
    
    image = _find(path)
    array = np.array(image)
        
    out = {}
    
    # Gaussian filter
    for size in gaussian_kernels:
        std = 0.6*((size-1)*0.5 - 1) + 1
        gaussian = cv2.GaussianBlur(array, (size, size), sigmaX=std, sigmaY=std)
        gaussian = Image.fromarray(gaussian)
        id_gaussian = 'gaussian_filter_' + str(size) + 'x' + str(size)
        out[id_gaussian] = gaussian
    
    # Median filter
    for size in median_kernels:
        # cv2 and PIL implementations are completely equivalent but cv2 is
        # much faster
        median = cv2.medianBlur(array, size)
        median = Image.fromarray(median)
        id_median = 'median_filter_' + str(size) + 'x' + str(size)
        out[id_median] = median
    
    return out


def compression_attack(path, compression_quality_factors=(10, 50, 90), **kwargs):
    """
    Generates jpeg compressed versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    compression_quality_factors : Tuple, optional
        All of the desired compression qualities. The default is (10, 50, 90).

    Returns
    -------
    out : Dictionary
        Compressed variations (as PIL images)
        
    """
    
    image = _find(path)
        
    out = {}
    
    for factor in compression_quality_factors:
        id_factor = 'jpg_compression_' + str(factor)
        
        # Trick to compress using jpg without actually saving to disk
        with BytesIO() as f:
            image.save(f, format='JPEG', quality=factor)
            f.seek(0)
            img = Image.open(f)
            img.load()
            out[id_factor] = img
        
    return out


def scaling_attack(path, scaling_ratios=(0.4, 0.8, 1.2, 1.6), **kwargs):
    """
    Generates rescaled versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    scaling_ratios : Tuple, optional
        All of the desired scaling ratios. The default is (0.4, 0.8, 1.2, 1.6).

    Returns
    -------
    out : Dictionary
        Scaled variations (as PIL images).

    """
    
    image = _find(path)
        
    width, height = image.size
    out = {}

    for ratio in scaling_ratios:
        id_ratio = 'scaling_' + str(ratio)
        out[id_ratio] = image.resize((round(ratio*width), round(ratio*height)),
                                     resample=Image.BICUBIC)
    
    return out


def cropping_attack(path, cropping_percentages=(5, 10, 20, 40, 60),
                    resize_cropping=True, **kwargs):
    """
    Generates cropped versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    cropping_percentages : Tuple, optional
        Desired cropped percentages (5 means we crop 5% of the image). We crop
        from the center of the image. The default is (5, 10, 20, 40, 60).
    resize_cropping : Boolean, optional
        Indicates if cropped images should be resized to original size or
        not. The default is True.

    Returns
    -------
    out : Dictionary
        Cropped variations (as PIL images)

    """
    
    image = _find(path)
        
    width, height = image.size
    out = {}
    
    for percentage in cropping_percentages:
        id_crop = 'cropping_' + str(percentage)
        ratio = 1 - percentage/100
        midx = width//2 + 1
        midy = height//2 + 1
        new_width = ratio*width
        new_height = ratio*height
        left = midx - new_width//2 - 1
        right = midx + new_width//2
        top = midy - new_height//2 - 1
        bottom = midy + new_height//2
        
        cropped = image.crop((left, top, right, bottom))
        
        if (resize_cropping):
            cropped = cropped.resize((width, height), resample=Image.BICUBIC)
            id_crop += '_and_rescaling'
            
        out[id_crop] = cropped
        
    return out
          
    
def rotation_attack(path, rotation_angles=(5, 10, 20, 40, 60),
                    resize_rotation=True, **kwargs):
    """
    Generates rotated versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    rotation_angles : Tuple, optional
        Desired angles of rotation (in degrees counter-clockwise).
        The default is (5, 10, 20, 40, 60).
    resize_rotation : Boolean, optional
        Indicates if rotated images including the boundary zone should be
        resized to original size ot not. The default is True.

    Returns
    -------
    out : Dictionary
        Rotated variations (as PIL images).

    """
    
    image = _find(path)
        
    size = image.size
    out = {}

    for angle in rotation_angles:
        id_angle = 'rotation_' + str(angle) 
        rotated = image.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        if (resize_rotation):
            rotated = rotated.resize(size, resample=Image.BICUBIC)
            id_angle += '_and_rescaling'
            
        out[id_angle] = rotated
    
    return out


def shearing_attack(path, shearing_angles=(1, 2, 5, 10, 20), **kwargs):
    """
    Generates sheared versions of the original image, in both directions.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    shearing_angles : Tuple, optional
        Desired shear angles (in degrees counter-clockwise). The default
        is (1, 2, 5, 10, 20).

    Returns
    -------
    out : Dictionary
        Sheared variations (as PIL images).

    """
    
    image = _find(path)
    
    out = {}
    
    for shear in shearing_angles:
        id_shear = 'shearing_' + str(shear)
        
        # use -shear to shear counter-clockwise, to be consistent with rotation
        sheared = F.affine(image, angle=0, translate=(0,0), scale=1, shear=(-shear, -shear),
                             interpolation=F.InterpolationMode.BICUBIC)
        
        out[id_shear] = sheared
        
    return out


def contrast_attack(path, contrast_factors=(0.5, 0.66, 1.5, 2), **kwargs):
    """
    Generates contrast changed versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    contrast_factors : Tuple, optional
        Desired enhancement factors. The default is (0.6, 0.8, 1.2, 1.4).

    Returns
    -------
    out : Dictionary
        Contrast changed images (as PIL images).

    """
    
    image = _find(path)
        
    enhancer = ImageEnhance.Contrast(image)
    
    out = {}
    
    for factor in contrast_factors:
        enhanced = enhancer.enhance(factor)
        id_enhanced = 'contrast_enhancement_' + str(factor)
        out[id_enhanced] = enhanced
        
    return out


def color_attack(path, color_factors=(0.5, 0.66, 1.5, 2), **kwargs):
    """
    Generates color changed versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    color_factors : Tuple, optional
        Desired enhancement factors. The default is (0.6, 0.8, 1.2, 1.4).

    Returns
    -------
    out : Dictionary
        Color changed images (as PIL images).

    """
    
    image = _find(path)
        
    enhancer = ImageEnhance.Color(image)
    
    out = {}
    
    for factor in color_factors:
        enhanced = enhancer.enhance(factor)
        id_enhanced = 'color_enhancement_' + str(factor)
        out[id_enhanced] = enhanced
        
    return out


def brightness_attack(path, brightness_factors=(0.5, 0.66, 1.5, 2), **kwargs):
    """
    Generates brightness changed versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    brightness_factors : Tuple, optional
        Desired enhancement factors. The default is (0.6, 0.8, 1.2, 1.4).

    Returns
    -------
    out : Dictionary
        Brightness changed images (as PIL images).

    """
    
    image = _find(path)
        
    enhancer = ImageEnhance.Brightness(image)
    
    out = {}
    
    for factor in brightness_factors:
        enhanced = enhancer.enhance(factor)
        id_enhanced = 'brightness_enhancement_' + str(factor)
        out[id_enhanced] = enhanced
        
    return out


def sharpness_attack(path, sharpness_factors=(0.5, 0.66, 1.5, 2), **kwargs):
    """
    Generates sharpness changed versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    sharpness_factors : Tuple, optional
        Desired enhancement factors. The default is (0.6, 0.8, 1.2, 1.4).

    Returns
    -------
    out : Dictionary
        Sharpness changed images (as PIL images).

    """
    
    image = _find(path)
        
    enhancer = ImageEnhance.Sharpness(image)
    
    out = {}
    
    for factor in sharpness_factors:
        enhanced = enhancer.enhance(factor)
        id_enhanced = 'sharpness_enhancement_' + str(factor)
        out[id_enhanced] = enhanced
        
    return out


def text_attack(path, text_lengths=(10, 20, 30, 40, 50), **kwargs):
    """
    Generates random text at random position in the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    text_lengths : Tuple, optional
        Length (number of character) of the added text. The default
        is (10, 20, 30, 40, 50).

    Returns
    -------
    out : Dictionary
        Text added image (as PIL images).

    """
    
    image = _find(path)
    width, height = image.size
    
    # List of characters for random strings
    characters = list(string.ascii_uppercase) + list(string.ascii_lowercase) + \
        list(string.digits)
    # Get a font. The size is calculated so that it is 40 for a 512 by 512
    # image, and changes linearly from this reference, so it always
    # takes the same relative space on different images
    if min(width, height) >= 100:
        size = round(40*width/512)
    else:
        size = round(6*width/100)
        
    if width/height <= 0.5 or  width/height >= 2:
        
        if min(width, height) >= 100:
            size = round(40*(width+height)/2/512) - 2
        else:
            size = round(6*(width+height)/2/100) - 2
    
    font = ImageFont.truetype('generator/Impact.ttf', size)
    
    out = {}

    for length in text_lengths:
        
        img = image.copy()
        # get a drawing context
        context = ImageDraw.Draw(img)
        
        # Random string generation
        sentence = ''.join(np.random.choice(characters, size=length,
                                            replace=True))
        
        # insert a newline every 20 characters for a 512-width image, and less
        # if the image is really long and not wide
        space = 20
        if height >= 2*width:
            space = int(np.floor(20*width/height))
        sentence = '\n'.join(sentence[i:i+space] for i in range(0, len(sentence), space))
        
        # Get the width and height of the text box
        dims = context.multiline_textbbox((0,0), sentence, font=font,
                                          stroke_width=2)
        width_text = dims[2] - dims[0]
        height_text = dims[3] - dims[1]
        
        # Random text position making sure that all text fits
        x = np.random.randint(1, width-width_text-1)
        y = np.random.randint(1, height-height_text-1)
        
        # Compute either white text black edegs or inverse based on mean
        #mean = np.mean(np.array(im)[x:x+w+1, y:y+h+1, :])
        #if mean <= 3*255/4:
        #    color = 'white'
        #    stroke = 'black'
        #else:
        #    color = 'black'
        #    stroke = 'white'
        
        color = 'white'
        stroke = 'black'

        context.multiline_text((x, y), sentence, font=font, fill=color,
                               stroke_fill=stroke, stroke_width=2, align='center')
        id_text = 'text_length_' + str(length)
        out[id_text] = img
        
    return out


# =============================================================================
# =============================================================================
# =============================================================================


# Define the legal arguments for all the attacks functions
VALID = ['gaussian_variances', 'speckle_variances', 'salt_pepper_amounts',
         'gaussian_kernels', 'median_kernels', 'compression_quality_factors',
         'scaling_ratios', 'cropping_percentages', 'resize_cropping',
         'rotation_angles', 'resize_rotation', 'shearing_angles',
         'contrast_factors', 'color_factors', 'brightness_factors',
         'sharpness_factors', 'text_lengths']


def perform_all_attacks(path, **kwargs):
    """
    Perform all of the attacks on a given image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    **kwargs : Named attack arguments.
        All of the attack parameters. Valid names are :
        'gaussian_variances', 'speckle_variances', 'salt_pepper_amounts',
        'gaussian_kernels', 'median_kernels', 'compression_quality_factors',
        'scaling_ratios', 'cropping_percentages', 'resize_cropping',
        'rotation_angles', 'resize_rotation', 'shearing_angles',
        'contrast_factors', 'color_factors', 'brightness_factors',
        'sharpness_factors', 'text_lengths'

    Raises
    ------
    TypeError
        If one of the name arguments is not valid.

    Returns
    -------
    out : Dictionary
        All attacked versions of the given image (as PIL images).

    """
    
    for arg in kwargs.keys():
        if (arg not in VALID):
            raise TypeError('Unexpected keyword argument \'' + arg + '\'')
            
    # Apply all attacks and merge
    out = noise_attack(path, **kwargs)
    out = {**out, **filter_attack(path, **kwargs)}
    out = {**out, **compression_attack(path, **kwargs)}
    out = {**out, **scaling_attack(path, **kwargs)}
    out = {**out, **cropping_attack(path, **kwargs)}
    out = {**out, **rotation_attack(path, **kwargs)}
    out = {**out, **shearing_attack(path, **kwargs)}
    out = {**out, **contrast_attack(path, **kwargs)}
    out = {**out, **color_attack(path, **kwargs)}
    out = {**out, **brightness_attack(path, **kwargs)}
    out = {**out, **sharpness_attack(path, **kwargs)}
    out = {**out, **text_attack(path, **kwargs)}
    
    return out


def save_attack(attacks, save_name, extension='PNG'):
    """
    Save the result of one (or multiple) attack on disk.

    Parameters
    ----------
    attacks : Dictionary
        Contains the attacked images (as returned by an attack function)
    save_name : str
        the prefix name to save the files (they will be saved as
        save_name_attack_id.format for example, where attack_id is the name
        of the given attack)
    extension : str, optional
        Format used to save the images (png by default as it is not lossy).
        The default is 'PNG'.

    Returns
    -------
    None.

    """

    for key in attacks.keys():
        name = save_name + '_' + key + '.' + extension.lower()
        attacks[key].save(name, format=extension)


def perform_all_and_save(path, save_name, extension='PNG', **kwargs):
    """
    Perform all of the attacks on a given image and save them on disk.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    save_name : str
        the prefix name to save the files (they will be saved as
        save_name_attack_id.format for example, where attack_id is the name
        of the given attack)
    extension : str, optional
        Format used to save the images (png by default as it is not lossy).
        The default is 'PNG'.
    **kwargs : Named attack arguments.
        All of the attack parameters. Valid names are :
        'gaussian_variances', 'speckle_variances', 'salt_pepper_amounts',
        'gaussian_kernels', 'median_kernels', 'compression_quality_factors',
        'scaling_ratios', 'cropping_percentages', 'resize_cropping',
        'rotation_angles', 'resize_rotation', 'shearing_angles',
        'contrast_factors', 'color_factors', 'brightness_factors',
        'sharpness_factors', 'text_lengths'

    Returns
    -------
    None.

    """
    
    attacks = perform_all_attacks(path, **kwargs)
    save_attack(attacks, save_name, extension)
    
    
def perform_all_and_save_list(path_list, save_name_list=None, extension='PNG',
                              **kwargs):
    """
    Perform all of the attacks on all images in a list and save them on disk.

    Parameters
    ----------
    path_list : array of str
        Path names to the images.
    save_name_list : array of str, optional
        the prefix names to save the files (they will be saved as 
        save_name_attack_id.format for example, where attack_id is the name of
        the given attack). If not given, will default to path_list names assuming
        that the last dot (.) is for the extension. The default is None.
    extension : str, optional
        Format used to save the images (png by default as it is not lossy).
        The default is 'PNG'.
    **kwargs : Named attack arguments.
        All of the attack parameters. Valid names are :
        'gaussian_variances', 'speckle_variances', 'salt_pepper_amounts',
        'gaussian_kernels', 'median_kernels', 'compression_quality_factors',
        'scaling_ratios', 'cropping_percentages', 'resize_cropping',
        'rotation_angles', 'resize_rotation', 'shearing_angles',
        'contrast_factors', 'color_factors', 'brightness_factors',
        'sharpness_factors', 'text_lengths'

    Returns
    -------
    None.

    """
    
    if save_name_list is None:
        save_name_list = [name.rsplit('.', 1)[0] for name in path_list]
    
    for path, save_name in tqdm(zip(path_list, save_name_list), 
                                total=len(path_list)):
        perform_all_and_save(path, save_name, **kwargs)
        
        
def retrieve_ids(**kwargs):
    """
    Retrieves the IDs of the attacks performed with the parameters in
    kwargs. This is useful to compute the ROC curves for each attack separately
    later on.

    Parameters
    ----------
    **kwargs : Named attack arguments.
        All of the attack parameters. Valid names are :
        'gaussian_variances', 'speckle_variances', 'salt_pepper_amounts',
        'gaussian_kernels', 'median_kernels', 'compression_quality_factors',
        'scaling_ratios', 'cropping_percentages', 'resize_cropping',
        'rotation_angles', 'resize_rotation', 'shearing_angles',
        'contrast_factors', 'color_factors', 'brightness_factors',
        'sharpness_factors', 'text_lengths'

    Raises
    ------
    TypeError
        If one of the name arguments is not valid.

    Returns
    -------
    IDs : List
        The IDs of the attacks.

    """
    
    # First check that all arguments are valids
    for arg in kwargs.keys():
        if (arg not in VALID):
            raise TypeError('Unexpected keyword argument \'' + arg + '\'')
            
    # Initialize output
    IDs = []
    
    # 
    if ('resize_cropping' in kwargs.keys()):
        resize_cropping = kwargs['resize_cropping']
    else:
        resize_cropping = True
    
    if ('resize_rotation' in kwargs.keys()):
        resize_rotation = kwargs['resize_rotation']
    else:
        resize_rotation = True
    
    # Convenient wrapper function to use for switch
    def wrapper(format_):
        def add_to_IDS(array):
            for value in array:
                IDs.append(format_.format(val=value))
        return add_to_IDS
    
    # Switch dictionary
    switch_dic = {
        'gaussian_variances': wrapper('gaussian_noise_{val}'),
        'speckle_variances': wrapper('speckle_noise_{val}'),
        'salt_pepper_amounts': wrapper('s&p_noise_{val}'),
        'gaussian_kernels': wrapper('gaussian_filter_{val}x{val}'),
        'median_kernels': wrapper('median_filter_{val}x{val}'),
        'compression_quality_factors': wrapper('jpg_compression_{val}'),
        'scaling_ratios': wrapper('scaling_{val}'),
        'cropping_percentages': wrapper('cropping_{val}_and_rescaling') if \
            resize_cropping else wrapper('cropping_{val}'),
        'rotation_angles': wrapper('rotation_{val}_and_rescaling') if \
            resize_rotation else wrapper('rotation_{val}'),
        'shearing_angles': wrapper('shearing_{val}'),
        'contrast_factors': wrapper('contrast_enhancement_{val}'),
        'color_factors': wrapper('color_enhancement_{val}')   ,
        'brightness_factors': wrapper('brightness_enhancement_{val}'),
        'sharpness_factors': wrapper('sharpness_enhancement_{val}'),
        'text_lengths': wrapper('text_length_{val}')
        }
    
    # Performs the switch on the intersection of keys (not only on kwargs.keys()
    # because the keys may contains e.g `resize_cropping` which is not included on
    # the switch and thus would raise an error)
    for key in (kwargs.keys() & switch_dic.keys()):
        switch_dic[key](kwargs[key])
    
    return IDs

        