#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:39:26 2021

@author: cyrilvallez
"""

# =============================================================================
# Some defaults parameters for better plots, matching the font and sizes of
# latex reports
# =============================================================================

import os
import matplotlib.pyplot as plt

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin/'

plt.rcParams['text.latex.preamble'] = r"""
\usepackage{mathtools}
\usepackage{amsmath}
"""
# Can contain other packages


plt.rc('font', family=['serif'])
plt.rc('font', serif=['Computer Modern Roman'])
plt.rc('savefig', dpi=400)
plt.rc('savefig', bbox='tight')
plt.rc('savefig', format='pdf')
plt.rc('figure', dpi=100)
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=16)
plt.rc('lines', linewidth=3.5)
plt.rc('lines', markersize=9)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

#### print(plt.rcParams) FOR A FULL LIST OF PARAMETERS
