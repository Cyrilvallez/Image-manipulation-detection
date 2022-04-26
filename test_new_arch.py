#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:53:19 2022

@author: cyrilvallez
"""

import torch
from torchvision import transforms as T
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