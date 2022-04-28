#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:53:19 2022

@author: cyrilvallez
"""

import torch
from torchvision import transforms as T
import torch.nn as nn
from PIL import Image

image1 = Image.open('Datasets/ILSVRC2012_img_val/Experimental/ILSVRC2012_val_00000018.JPEG')
image2 = Image.open('Datasets/ILSVRC2012_img_val/Experimental/ILSVRC2012_val_00000022.JPEG')

transforms = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

images = torch.stack((transforms(image1), transforms(image2)), dim=0)
#
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', verbose=False)

@torch.no_grad()
def extract_features(images):

    # feats = self.get_intermediate_layers(images, n=1)[0].clone()

    cls_output_token = feats[:, 0, :]  #  [CLS] token
    # GeM with exponent 4 for output patch tokens
    b, h, w, d = len(images), int(images.shape[-2] / model.patch_embed.patch_size), int(images.shape[-1] / model.patch_embed.patch_size), feats.shape[-1]
    feats = feats[:, 1:, :].reshape(b, h, w, d)
    feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
    feats = nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)
    # concatenate [CLS] token and GeM pooled patch tokens
    feats = torch.cat((cls_output_token, feats), dim=1)

    return feats  
        
# model.forward = extract_features  

# test = model(images)
      

#%%

def hack(model):
    @torch.no_grad()
    def extract_features(images):

        feats = model.get_intermediate_layers(images, n=1)[0].clone()

        cls_output_token = feats[:, 0, :]  #  [CLS] token
        # GeM with exponent 4 for output patch tokens
        b, h, w, d = len(images), int(images.shape[-2] / model.patch_embed.patch_size), int(images.shape[-1] / model.patch_embed.patch_size), feats.shape[-1]
        feats = feats[:, 1:, :].reshape(b, h, w, d)
        feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
        feats = nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)
        # concatenate [CLS] token and GeM pooled patch tokens
        feats = torch.cat((cls_output_token, feats), dim=1)

        return feats  
    
    return extract_features

model.forward = hack(model)


test = model(images)     
        
        