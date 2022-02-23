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

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
from skimage import util, transform
import string
from io import BytesIO
from tqdm import tqdm
np.random.seed(256)

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
    
    if (type(path) == str):
        image = Image.open(path)
    else:
        image = path
        
    return image

# =============================================================================
# =============================================================================
# ================================ ATTACKS =====================================


def noise_attack(path, gaussian_var=(0.01, 0.02, 0.05), speckle_var=(0.01, 0.02, 0.05),
                 sp_amount=(0.05, 0.1, 0.15), **kwargs):
    """
    Generates noisy versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    gaussian_var : Tuple, optional
        Variances for the Gaussian noise. The default is (0.01, 0.02, 0.05).
    speckle_var : Tuple, optional
        Variances for the Speckle noise. The default is (0.01, 0.02, 0.05).
    sp_amount : Tuple, optional
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
    for var in gaussian_var:
        gaussian = util.random_noise(array, mode='gaussian', mean=0, var=var)
        gaussian = util.img_as_ubyte(gaussian)
        gaussian = Image.fromarray(gaussian)
        id_gaussian = 'gaussian_noise_' + str(var)
        out[id_gaussian] = gaussian
    
    # Salt and pepper noise
    for amount in sp_amount:
        sp = util.random_noise(array, mode='s&p', amount=amount)
        sp = util.img_as_ubyte(sp)
        sp = Image.fromarray(sp)
        id_sp = 's&p_noise_' + str(amount)
        out[id_sp] = sp
      
    # Speckle noise 
    for var in speckle_var:
        speckle = util.random_noise(array, mode='speckle', mean=0, var=var)
        speckle = util.img_as_ubyte(speckle)
        speckle = Image.fromarray(speckle)
        id_speckle = 'speckle_noise_' + str(var)
        out[id_speckle] = speckle
    
    return out


def filter_attack(path, gaussian_kernel=(1, 2, 3),
                  median_kernel=(3, 5, 7), **kwargs):
    """
    Generates filtered versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    gaussian_kernel : Tuple, optional
        Sizes for the gaussian filter in one direction, from the CENTER pixel
        (thus a size of 1 gives a 3x3 filter, 2 gives 5x5 filter etc).
        The default is (1, 2, 3).
    median_kernel : Tuple, optional
        Sizes for the median filter (true size, give 3 for a 3x3 filter).
        The default is (3, 5, 7).

    Returns
    -------
    out : Dictionary
        Filtered variations if the image (as PIL images)

    """
    
    image = _find(path)
        
    out = {}
    
    # Gaussian filter
    for size in gaussian_kernel:
        gaussian = image.filter(ImageFilter.GaussianBlur(radius=size))
        g_size = str(2*size+1)
        id_gaussian = 'gaussian_filter_' + g_size + 'x' + g_size
        out[id_gaussian] = gaussian
    
    # Median filter
    for size in median_kernel:
        median = image.filter(ImageFilter.MedianFilter(size=size))
        id_median = 'median_filter_' + str(size) + 'x' + str(size)
        out[id_median] = median
    
    return out


def compression_attack(path, quality_factors=(10, 50, 90), **kwargs):
    """
    Generates jpeg compressed versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    quality_factors : Tuple, optional
        All of the desired compression qualities. The default is (10, 50, 90).

    Returns
    -------
    out : Dictionary
        Compressed variations (as PIL images)
        
    """
    
    image = _find(path)
        
    out = {}
    
    for factor in quality_factors:
        id_factor = 'compressed_jpg_' + str(factor)
        
        # Trick to compress using jpg without actually saving to disk
        with BytesIO() as f:
            image.save(f, format='JPEG', quality=factor)
            f.seek(0)
            img = Image.open(f)
            img.load()
            out[id_factor] = img
        
    return out


def scaling_attack(path, ratios=(0.4, 0.8, 1.2, 1.6), **kwargs):
    """
    Generates rescaled versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    ratios : Tuple, optional
        All of the desired scaling ratios. The default is (0.4, 0.8, 1.2, 1.6).

    Returns
    -------
    out : Dictionary
        Scaled variations (as PIL images).

    """
    
    image = _find(path)
        
    width, height = image.size
    out = {}

    for ratio in ratios:
        id_ratio = 'scaled_' + str(ratio)
        out[id_ratio] = image.resize((round(ratio*width), round(ratio*height)),
                                     resample=Image.LANCZOS)
    
    return out


def cropping_attack(path, percentages=(5, 10, 20, 40, 60), resize_crop=True,
                    **kwargs):
    """
    Generates cropped versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    percentages : Tuple, optional
        Desired cropped percentages (5 means we crop 5% of the image). We crop
        from the center of the image. The default is (5, 10, 20, 40, 60).
    resize_crop : Boolean, optional
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
    
    for percentage in percentages:
        id_crop = 'cropped_' + str(percentage)
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
        
        if (resize_crop):
            cropped = cropped.resize((width, height))
            id_crop += '_resized'
            
        out[id_crop] = cropped
        
    return out
            
    
def rotation_attack(path, angles_rot=(5, 10, 20, 40, 60), resize_rot=True,
                    **kwargs):
    """
    Generates rotated versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    angles_rot : Tuple, optional
        Desired angles of rotation (in degrees counter-clockwise).
        The default is (5, 10, 20, 40, 60).
    resize_rot : Boolean, optional
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

    for angle in angles_rot:
        id_angle = 'rotated_' + str(angle) 
        rotated = image.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        if (resize_rot):
            rotated = rotated.resize(size, resample=Image.LANCZOS)
            id_angle += '_resized'
            
        out[id_angle] = rotated
    
    return out


def shearing_attack(path, angles_shear=(1, 2, 5, 10, 20), **kwargs):
    """
    Generates sheared versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    angles_shear : Tuple, optional
        Desired shear angles (in degrees counter-clockwise). The default
        is (1, 2, 5, 10, 20).

    Returns
    -------
    out : Dictionary
        Sheared variations (as PIL images).

    """
    
    image = _find(path)
    
    array = np.array(image)
    
    # trsnform the angles in radians
    angles_rad = np.pi/180*np.array(angles_shear)
    
    out = {}
    
    for angle, degree in zip(angles_rad, angles_shear):
        id_shear = 'sheared_' + str(degree)
        
        transfo = transform.AffineTransform(shear=angle)
        sheared = transform.warp(array, transfo, order=4)
        sheared = util.img_as_ubyte(sheared)
        sheared = Image.fromarray(sheared)
        
        out[id_shear] = sheared
        
    return out


def contrast_attack(path, factors_contrast=(0.6, 0.8, 1.2, 1.4), **kwargs):
    """
    Generates contrast changed versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    factors_contrast : Tuple, optional
        Desired enhancement factors. The default is (0.6, 0.8, 1.2, 1.4).

    Returns
    -------
    out : Dictionary
        Contrast changed images (as PIL images).

    """
    
    image = _find(path)
        
    enhancer = ImageEnhance.Contrast(image)
    
    out = {}
    
    for factor in factors_contrast:
        enhanced = enhancer.enhance(factor)
        id_enhanced = 'contrast_enhanced_' + str(factor)
        out[id_enhanced] = enhanced
        
    return out


def color_attack(path, factors_color=(0.6, 0.8, 1.2, 1.4), **kwargs):
    """
    Generates color changed versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    factors_color : Tuple, optional
        Desired enhancement factors. The default is (0.6, 0.8, 1.2, 1.4).

    Returns
    -------
    out : Dictionary
        Color changed images (as PIL images).

    """
    
    image = _find(path)
        
    enhancer = ImageEnhance.Color(image)
    
    out = {}
    
    for factor in factors_color:
        enhanced = enhancer.enhance(factor)
        id_enhanced = 'color_enhanced_' + str(factor)
        out[id_enhanced] = enhanced
        
    return out


def brightness_attack(path, factors_bright=(0.6, 0.8, 1.2, 1.4), **kwargs):
    """
    Generates brightness changed versions of the original image

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    factors_bright : Tuple, optional
        Desired enhancement factors. The default is (0.6, 0.8, 1.2, 1.4).

    Returns
    -------
    out : Dictionary
        Brightness changed images (as PIL images).

    """
    
    image = _find(path)
        
    enhancer = ImageEnhance.Brightness(image)
    
    out = {}
    
    for factor in factors_bright:
        enhanced = enhancer.enhance(factor)
        id_enhanced = 'brightness_enhanced_' + str(factor)
        out[id_enhanced] = enhanced
        
    return out


def sharpness_attack(path, factors_sharp=(0.6, 0.8, 1.2, 1.4), **kwargs):
    """
    Generates sharpness changed versions of the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    factors_sharp : Tuple, optional
        Desired enhancement factors. The default is (0.6, 0.8, 1.2, 1.4).

    Returns
    -------
    out : Dictionary
        Sharpness changed images (as PIL images).

    """
    
    image = _find(path)
        
    enhancer = ImageEnhance.Sharpness(image)
    
    out = {}
    
    for factor in factors_sharp:
        enhanced = enhancer.enhance(factor)
        id_enhanced = 'sharpness_enhanced_' + str(factor)
        out[id_enhanced] = enhanced
        
    return out


def text_attack(path, lengths=(10, 20, 30, 40, 50), **kwargs):
    """
    Generates random text at random position in the original image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    lengths : Tuple, optional
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
    characters = list(string.ascii_lowercase + string.ascii_uppercase \
                      + string.digits + ' ')
    # Get a font. The size is calculated so that it is 40 for a 512-width
    # image, and changes linearly from this reference, so it always
    # takes the same relative horizontal space on different images
    font = ImageFont.truetype('Impact.ttf', round(40*width/512))
    
    out = {}
        
    for length in lengths:
        
        img = image.copy()
        # get a drawing context
        context = ImageDraw.Draw(img)
        
        # Random string generation
        sentence = ''.join(np.random.choice(characters, size=length,
                                            replace=True))
        # insert a newline every 20 characters
        for i in range(len(sentence)//20):
            sentence = sentence[0:20*(i+1)+i] + '\n' + \
                sentence[20*(i+1)+1:]
        
        # Get the width and height of the text box
        dims = context.multiline_textbbox((0,0), sentence, font=font,
                                          stroke_width=2)
        width_text = dims[2] - dims[0]
        height_text = dims[3] - dims[1]
        
        # Random text position making sure that all text fits
        x = np.random.randint(10, width-width_text-10)
        y = np.random.randint(10, height-height_text-10)
        
        # Compute either white text back edegs or inverse based on mean
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
        id_text = 'text_' + str(length)
        out[id_text] = img
        
    return out


# =============================================================================
# =============================================================================
# =============================================================================


# Define the legal arguments for all the attacks functions
VALID = ['gaussian_var', 'speckle_var', 'sp_amount', 'gaussian_kernel', 'median_kernel',
         'quality_factors', 'ratios', 'percentages', 'resize_crop', 'angles_rot',
         'resize_rot', 'angles_shear', 'factors_contrast', 'factors_color',
         'factors_bright', 'factors_sharp', 'lengths']


def perform_all_attacks(path, **kwargs):
    """
    Perform all of the attacks on a given image.

    Parameters
    ----------
    path : PIL image or str
        PIL image or path to the image.
    **kwargs : Named attack arguments.
        All of the attack parameters. Valid names are :
        'gaussian_var', 'speckle_var', 'sp_amount', 'gaussian_kernel', 'median_kernel',
        'quality_factors', 'ratios', 'percentages', 'resize_crop', 'angles_rot',
        'resize_rot', 'angles_shear', 'factors_contrast', 'factors_color',
        'factors_bright', 'factors_sharp', 'lengths'

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
        'gaussian_var', 'speckle_var', 'sp_amount', 'gaussian_kernel', 'median_kernel',
        'quality_factors', 'ratios', 'percentages', 'resize_crop', 'angles_rot',
        'resize_rot', 'angles_shear', 'factors_contrast', 'factors_color',
        'factors_bright', 'factors_sharp', 'lengths'

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
        'gaussian_var', 'speckle_var', 'sp_amount', 'gaussian_kernel', 'median_kernel',
        'quality_factors', 'ratios', 'percentages', 'resize_crop', 'angles_rot',
        'resize_rot', 'angles_shear', 'factors_contrast', 'factors_color',
        'factors_bright', 'factors_sharp', 'lengths'

    Returns
    -------
    None.

    """
    
    if save_name_list is None:
        save_name_list = [name.rsplit('.', 1)[0] for name in path_list]
    
    for path, save_name in tqdm(zip(path_list, save_name_list)):
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
        'gaussian_var', 'speckle_var', 'sp_amount', 'gaussian_kernel', 'median_kernel',
        'quality_factors', 'ratios', 'percentages', 'resize_crop', 'angles_rot',
        'resize_rot', 'angles_shear', 'factors_contrast', 'factors_color',
        'factors_bright', 'factors_sharp', 'lengths'

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
    
    IDs = []
    
    # Loop again 
    for arg in kwargs.keys():
        
        # Not pretty but works. Just loop over all possible attacks.
        if (arg=='g_var'):
            for value in kwargs[arg]:
                IDs.append('gaussian_noise_' + str(value))
        elif (arg=='s_var'):
            for value in kwargs[arg]:
                IDs.append('speckle_noise_' + str(value))
        elif (arg=='sp_amount'):
            for value in kwargs[arg]:
                IDs.append('s&p_noise_' + str(value))
        elif (arg=='g_kernel'):
            for value in kwargs[arg]:
                g_size = str(2*value+1)
                IDs.append('gaussian_filter_' + g_size + 'x' + g_size)
        elif (arg=='m_kernel'):
            for value in kwargs[arg]:
                IDs.append('median_filter_' + str(value) + 'x' + str(value))
        elif (arg=='quality_factors'):
            for value in kwargs[arg]:
                IDs.append('compressed_jpg_' + str(value))
        elif (arg=='ratios'):
            for value in kwargs[arg]:
                IDs.append('scaled_' + str(value))
        elif (arg=='percentages'):
            for value in kwargs[arg]:
                if ('resize_crop' in kwargs.keys()):
                    if (kwargs['resize_crop']):
                        IDs.append('cropped_' + str(value) + '_resized')
                    else:
                        IDs.append('cropped_' + str(value))
                else:
                    IDs.append('cropped_' + str(value) + '_resized')    
        elif (arg=='angles_rot'):
            for value in kwargs[arg]:
                if ('resize_rot' in kwargs.keys()):
                    if (kwargs['resize_rot']):
                        IDs.append('rotated_' + str(value)  + '_resized')
                    else:
                        IDs.append('rotated_' + str(value))
                else:
                    IDs.append('rotated_' + str(value) + '_resized')       
        elif (arg=='angles_shear'):
            for value in kwargs[arg]:
                IDs.append('sheared_' + str(value))
        elif (arg=='factors_contrast'):
            for value in kwargs[arg]:
                IDs.append('contrast_enhanced_' + str(value))
        elif (arg=='factors_color'):
            for value in kwargs[arg]:
                IDs.append('color_enhanced_' + str(value))
        elif (arg=='factors_bright'):
            for value in kwargs[arg]:
                IDs.append('brightness_enhanced_' + str(value))
        elif (arg=='factors_sharp'):
            for value in kwargs[arg]:
                IDs.append('sharpness_enhanced_' + str(value))
        elif (arg=='lengths'):
            for value in kwargs[arg]:                           
                IDs.append('text_' + str(value))
    
    return IDs
            

        
        
        
        
        
        