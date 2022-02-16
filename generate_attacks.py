#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 08:29:00 2022

@author: cyrilvallez
"""

# =============================================================================
# This script generates numerous variations of an image, designed to test
# the robustness of different hashing algorithms
#
# With default argument, it takes ~8 s to perform and save all attacks on disk
# for 1 image
# Thus it should take ~1 h for 500 images as a reference
# =============================================================================

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
from skimage import util, transform
import string
from io import BytesIO

def _find(path):
    ''' Internal functions intended to return a PIL image (useful to limit 
        reuse of this snippet)
    - inputs : 
        path : PIL image or path to image file
    '''
    if (type(path) == str):
        im = Image.open(path)
    else:
        im = path
        
    return im

# =============================================================================
# =============================================================================
# ================================ ATTACKS =====================================


def noise_attack(path, g_var=[0.01, 0.02, 0.05], s_var=[0.01, 0.02, 0.05],
                 sp_amount=[0.05, 0.1, 0.15], **kwargs):
    ''' Generates noisy versions of the original image.
    - inputs : 
        path : PIL image or path to image file
        g_var : list of variances of gaussian noise
        s_var : list of variances of speckle noise
        sp_amount : list of amount of salt and pepper noise
        
    - outputs :
        noisy images variations (as PIL images)
    '''
    
    im = _find(path)
    array = np.array(im)
    
    out = {}
    
    # Gaussian noise 
    for var in g_var:
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
    for var in s_var:
        speckle = util.random_noise(array, mode='speckle', mean=0, var=var)
        speckle = util.img_as_ubyte(speckle)
        speckle = Image.fromarray(speckle)
        id_speckle = 'speckle_noise_' + str(var)
        out[id_speckle] = speckle
    
    return out


def filter_attack(path, g_kernel=[1, 2, 3], m_kernel=[3, 5, 7], **kwargs):
    ''' Generates filtered versions of the original image.
    - inputs : 
        path : PIL image or path to image file
        g_kernel : list of sizes for the gaussian filter in one direction,
                   from the CENTER pixel (thus a size of 1 gives a 3x3 filter,
                   2 gives 5x5 filter etc)
        m_kernel : list of sizes for the median filter (true size,
                   give 3 for a 3x3 filter)
        
    - outputs :
        filtered variations (as PIL images)
    '''
    
    im = _find(path)
        
    out = {}
    
    # Gaussian filter
    for size in g_kernel:
        gaussian = im.filter(ImageFilter.GaussianBlur(radius=size))
        g_size = str(2*size+1)
        id_gaussian = 'gaussian_filter_' + g_size + 'x' + g_size
        out[id_gaussian] = gaussian
    
    # Median filter
    for size in m_kernel:
        median = im.filter(ImageFilter.MedianFilter(size=size))
        id_median = 'median_filter_' + str(size) + 'x' + str(size)
        out[id_median] = median
    
    return out


def compression_attack(path, quality_factors=[10, 50, 90], **kwargs):
    ''' Generates jpeg compressed versions of the original image.
    - inputs : 
        path : PIL image or path to image file
        quality_factors : list of all the desired qualities
        
    - outputs :
        compressed variations (as PIL images)
    '''
    im = _find(path)
        
    out = {}
    
    for factor in quality_factors:
        id_factor = 'compressed_jpg_' + str(factor)
        
        # Trick to compress using jpg without actually saving to disk
        with BytesIO() as f:
            im.save(f, format='JPEG', quality=factor)
            f.seek(0)
            img = Image.open(f)
            img.load()
            out[id_factor] = img
        
    return out


def scaling_attack(path, ratios=[0.4, 0.8, 1.2, 1.6], **kwargs):
    ''' Generates rescaled versions of the original image.
    - inputs : 
        path : PIL image or path to image file
        ratios : list of all the desired scaling ratios
        
    - outputs :
        scaled variations (as PIL images)
    '''
    
    im = _find(path)
        
    width, height = im.size
    out = {}

    for ratio in ratios:
        id_ratio = 'scaled_' + str(ratio)
        out[id_ratio] = im.resize((round(ratio*width), round(ratio*height)),
                                     resample=Image.LANCZOS)
    
    return out


def cropping_attack(path, percentages=[5, 10, 20, 40, 60], resize_crop=True,
                    **kwargs):
    ''' Generates cropped versions of the original image
    - inputs : 
        path : PIL image or path to image file
        percentage : list of desired cropped percentages (5 means we crop
                     5% of the image)
        resize_crop : boolean indicating if cropped images should be resized to
                 original size ot not
                 
    - outputs :
        cropped variations (as PIL images)
    '''
    
    im = _find(path)
        
    width, height = im.size
    out = {}
    
    for percentage in percentages:
        id_crop = 'cropped_' + str(percentage)
        r = 1 - percentage/100
        midx = width//2 + 1
        midy = height//2 + 1
        w = r*width
        h = r*height
        left = midx - w//2 - 1
        right = midx + w//2
        top = midy - h//2 - 1
        bottom = midy + h//2
        
        cropped = im.crop((left, top, right, bottom))
        
        if (resize_crop):
            cropped = cropped.resize((width, height))
            id_crop += '_resized'
            
        out[id_crop] = cropped
        
    return out
            
    
def rotation_attack(path, angles_rot=[5, 10, 20, 40, 60], resize_rot=True,
                    **kwargs):
    ''' Generates rotated versions of the original image
    - inputs : 
        path : PIL image or path to image file
        angles_rot : list of desired angles of rotation (in degrees counter
                     clockwise)
        resize_rot : boolean indicating if rotated images including the boundary
                     zone should be resized to original size ot not
                 
    - outputs :
        rotated variations (as PIL images)
    '''
    
    im = _find(path)
        
    size = im.size
    out = {}

    for angle in angles_rot:
        id_angle = 'rotated_' + str(angle) 
        rotated = im.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        if (resize_rot):
            rotated = rotated.resize(size, resample=Image.LANCZOS)
            id_angle += '_resized'
            
        out[id_angle] = rotated
    
    return out


def shearing_attack(path, angles_shear=[1, 2, 5, 10, 20], **kwargs):
    ''' Generates sheared versions of the original image
    - inputs : 
        path : PIL image or path to image file
        angles_shear : list of desired shear angles (in degrees counter
                       clockwise)
                 
    - outputs :
        sheared variations (as PIL images)
    '''
    
    im = _find(path)
    
    array = np.array(im)
    
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


def contrast_attack(path, factors_contrast=[0.6, 0.8, 1.2, 1.4], **kwargs):
    ''' Generates contrast changed versions of the original image
    - inputs : 
        path : PIL image or path to image file
        factors_contrast : list of the enhancement factor
                 
    - outputs :
        contrast changed images (as PIL images)
    '''
    
    im = _find(path)
        
    enhancer = ImageEnhance.Contrast(im)
    
    out = {}
    
    for f in factors_contrast:
        enhanced = enhancer.enhance(f)
        id_enhanced = 'contrast_enhanced_' + str(f)
        out[id_enhanced] = enhanced
        
    return out


def color_attack(path, factors_color=[0.6, 0.8, 1.2, 1.4], **kwargs):
    ''' Generates color changed versions of the original image
    - inputs : 
        path : PIL image or path to image file
        factors_color : list of the enhancement factor
                 
    - outputs :
        color changed images (as PIL images)
    '''
    
    im = _find(path)
        
    enhancer = ImageEnhance.Color(im)
    
    out = {}
    
    for f in factors_color:
        enhanced = enhancer.enhance(f)
        id_enhanced = 'color_enhanced_' + str(f)
        out[id_enhanced] = enhanced
        
    return out


def brightness_attack(path, factors_bright=[0.6, 0.8, 1.2, 1.4], **kwargs):
    ''' Generates brightness changed versions of the original image
    - inputs : 
        path : PIL image or path to image file
        factors_bright : list of the enhancement factor
                 
    - outputs :
        brightness changed images (as PIL images)
    '''
    
    im = _find(path)
        
    enhancer = ImageEnhance.Brightness(im)
    
    out = {}
    
    for f in factors_bright:
        enhanced = enhancer.enhance(f)
        id_enhanced = 'brightness_enhanced_' + str(f)
        out[id_enhanced] = enhanced
        
    return out


def sharpness_attack(path, factors_sharp=[0.6, 0.8, 1.2, 1.4], **kwargs):
    ''' Generates sharpness changed versions of the original image
    - inputs : 
        path : PIL image or path to image file
        factors_sharp : list of the enhancement factor
                 
    - outputs :
        sharpness changed images (as PIL images)
    '''
    
    im = _find(path)
        
    enhancer = ImageEnhance.Sharpness(im)
    
    out = {}
    
    for f in factors_sharp:
        enhanced = enhancer.enhance(f)
        id_enhanced = 'sharpness_enhanced_' + str(f)
        out[id_enhanced] = enhanced
        
    return out


def text_attack(path, lengths=[10, 20, 30, 40, 50], **kwargs):
    ''' Generates random text at random position in the original image
    - inputs : 
        path : PIL image or path to image file
                 
    - outputs :
        text added image (as PIL images)
    '''
    
    im = _find(path)
    width, height = im.size
    
    # List of characters for random strings
    characters = list(string.ascii_lowercase + string.ascii_uppercase \
                      + string.digits + ' ')
    # Get a font. The size is calculated so that it is 30 for a 512-width
    # image, and changes linearly from this reference, so it always
    # takes the same relative horitontal space on different images
    font = ImageFont.truetype('Arial.ttf', round(30*width/512))
    # get a drawing context
    context = ImageDraw.Draw(im)
    
    out = {}
        
    for length in lengths:
        
        img = im.copy()
        # get a drawing context
        context = ImageDraw.Draw(img)
        
        # Random string generation
        sentence = ''.join(np.random.choice(characters, size=length,
                                            replace=True))
        # insert a newline in the middle
        sentence = sentence[0:len(sentence)//2] + '\n' + \
            sentence[len(sentence)//2:]
        # Random color generation
        color = tuple(np.random.randint(0, 256, 3))
        
        # Get the width and height of the text box
        dims = context.multiline_textbbox((0,0), sentence, font=font)
        w = dims[2] - dims[0]
        h = dims[3] - dims[1]
        
        # Random text position making sure that all text fits
        x = np.random.randint(0, width-w)
        y = np.random.randint(h+1, height-h)

        context.multiline_text((x, y), sentence, font=font, fill=color)
        id_text = 'text_' + str(length)
        out[id_text] = img
        
    return out


# =============================================================================
# =============================================================================
# =============================================================================


# Define the legal arguments for all the attacks functions
VALID = ['g_var', 's_var', 'sp_amount', 'g_kernel', 'm_kernel', 'quality_factors',
         'ratios', 'percentages', 'resize_crop', 'angles_rot', 'resize_rot',
         'angles_shear', 'factors_contrast', 'factors_color', 'factors_bright',
         'factors_sharp', 'lengths']


def perform_all_attacks(path, **kwargs):
    ''' Perform all of the attacks on a given image.
    - inputs : 
        path : PIL image or path to image file
                 
    - outputs :
        all attacked images (as PIL images)
    '''
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
    ''' Save the result of one (or multiple) attack on disk.
    - inputs : 
        attacks : Dictionary containing the attacked images (as returned by an
                  attack function)
        save_name : the prefix name to save the files (they will be saved as
                    save_name_attack_id.format for example, where attack_id
                    is the name of the given attack)
        extension : format used to save the images (png by default as it is not
                    lossy)
    '''
    
    for key in attacks.keys():
        name = save_name + '_' + key + '.' + extension.lower()
        attacks[key].save(name, format=extension)


def perform_all_and_save(path, save_name, extension='PNG', **kwargs):
    ''' Perform all of the attacks on a given image and save them on disk.
    - inputs : 
        path : PIL image or path to image file
        save_name : the prefix name to save the files (they will be saved as
                    save_name_attack_id.format for example, where attack_id
                    is the name of the given attack)
        extension : format used to save the images (png by default as it is not
                  lossy)
    '''
    
    attacks = perform_all_attacks(path, **kwargs)
    save_attack(attacks, save_name, extension)
        
        
        
        
        
        
        