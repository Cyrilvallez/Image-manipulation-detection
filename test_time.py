#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 08:23:30 2022

@author: cyrilvallez
"""

import generator
import time

path = 'test_hashing/BSDS500/Control_attacks/data26_shearing_2.png'

t0 = time.time()

for a in range(10):
    
    generator.perform_all_attacks(path, **generator.ATTACK_PARAMETERS)
    
dt = time.time() - t0

print(f'Time needed : {dt/10} s')