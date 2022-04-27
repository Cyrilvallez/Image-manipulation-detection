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

image = Image.open('Datasets/ILSVRC2012_img_val/Experimental/ILSVRC2012_val_00000018.JPEG')

transforms = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

image = transforms(image).unsqueeze(dim=0)

model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

test = model(image)

feats = model.get_intermediate_layers(image, n=1)[0].clone()



@torch.no_grad()
def extract_features(image, model):
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image = transform(image).unsqueeze(0)
    
    feats = model.get_intermediate_layers(image, n=1)[0].clone()

    cls_output_token = feats[:, 0, :]  #  [CLS] token
    # GeM with exponent 4 for output patch tokens
    b, h, w, d = len(image), int(image.shape[-2] / model.patch_embed.patch_size), int(image.shape[-1] / model.patch_embed.patch_size), feats.shape[-1]
    feats = feats[:, 1:, :].reshape(b, h, w, d)
    feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
    feats = nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)
    # concatenate [CLS] token and GeM pooled patch tokens
    feats = torch.cat((cls_output_token, feats), dim=1)

    return feats


# test = extract_features(image, model)

#%%

class A(object):
    
    def __init__(self):
        self.a = 10
        
    def method1(self):
        self.a += 5
        
    def method2(self):
        self.a += 1
        
        
# test = A()

test.method2 = test.method1

test.method2()

print(test.a)
        
        
        
        
        
        
        