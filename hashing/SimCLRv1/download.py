#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:51:28 2022

@author: cyrilvallez
"""

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import argparse
import os
from math import ceil
import requests
from tqdm import tqdm

from resnet_wider import resnet50x1, resnet50x2, resnet50x4

available_simclr_models = ['1x', '2x', '4x']
simclr_base_url = 'https://storage.googleapis.com/simclr-checkpoints/simclrv1/{category}/{model}/'
files = ['checkpoint', 'graph.pbtxt', 'model.ckpt-{category}.data-00000-of-00001',
         'model.ckpt-{category}.index', 'model.ckpt-{category}.meta']
simclr_categories = {
                     'finetune_100pct': {'1x':9384, '2x': 18768, '4x': 4692}, 
                     'finetune_10pct': {'1x':939, '2x': 939, '4x': 470}, 
                     'pretrain': {'1x':225206, '2x': 225206, '4x': 225206}
                     }
chunk_size = 1024 * 8

mapping = {'finetune_100pct': 'Finetuned_100pct', 'finetune_10pct': 'Finetuned_10pct', 
           'pretrain': 'Pretrained'}


def download(url, destination):
    if os.path.exists(destination):
        return
    response = requests.get(url, stream=True)
    save_response_content(response, destination)


def save_response_content(response, destination):
    if 'Content-length' in response.headers:
        total = int(ceil(int(response.headers['Content-length']) / chunk_size))
    else:
        total = None
    with open(destination, 'wb') as f:
        for data in tqdm(response.iter_content(chunk_size=chunk_size), leave=False, total=total):
            f.write(data)


def run_download(model, directory, simclr_category):

    url = simclr_base_url.format(model=model, category=simclr_category)
    model_category = simclr_categories[category][model]
    for file in tqdm(files):
        f = file.format(category=model_category)
        download(url + f, os.path.join(directory, f))


def convert(path, output_path):
    # 1. read tensorflow weight into a python dict
    vars_list = tf.train.list_variables(path)
    vars_list = [v[0] for v in vars_list]
    # print('#vars:', len(vars_list))

    sd = {}
    ckpt_reader = tf.train.load_checkpoint(path)
    for v in vars_list:
        sd[v] = ckpt_reader.get_tensor(v)

    sd.pop('global_step')

    # 2. convert the state_dict to PyTorch format
    conv_keys = [k for k in sd.keys() if k.split('/')[1].split('_')[0] == 'conv2d']
    conv_idx = []
    for k in conv_keys:
        mid = k.split('/')[1]
        if len(mid) == 6:
            conv_idx.append(0)
        else:
            conv_idx.append(int(mid[7:]))
    arg_idx = np.argsort(conv_idx)
    conv_keys = [conv_keys[idx] for idx in arg_idx]

    bn_keys = list(set([k.split('/')[1] for k in sd.keys() if k.split('/')[1].split('_')[0] == 'batch']))
    bn_idx = []
    for k in bn_keys:
        if len(k.split('_')) == 2:
            bn_idx.append(0)
        else:
            bn_idx.append(int(k.split('_')[2]))
    arg_idx = np.argsort(bn_idx)
    bn_keys = [bn_keys[idx] for idx in arg_idx]

    if '1x' in path:
        model = resnet50x1()
    elif '2x' in path:
        model = resnet50x2()
    elif '4x' in path:
        model = resnet50x4()
    else:
        raise NotImplementedError

    conv_op = []
    bn_op = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_op.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bn_op.append(m)

    for i_conv in range(len(conv_keys)):
        m = conv_op[i_conv]
        # assert the weight of conv has the same shape
        assert torch.from_numpy(sd[conv_keys[i_conv]]).permute(3, 2, 0, 1).shape == m.weight.data.shape
        m.weight.data = torch.from_numpy(sd[conv_keys[i_conv]]).permute(3, 2, 0, 1)

    for i_bn in range(len(bn_keys)):
        m = bn_op[i_bn]
        m.weight.data = torch.from_numpy(sd['base_model/' + bn_keys[i_bn] + '/gamma'])
        m.bias.data = torch.from_numpy(sd['base_model/' + bn_keys[i_bn] + '/beta'])
        m.running_mean = torch.from_numpy(sd['base_model/' + bn_keys[i_bn] + '/moving_mean'])
        m.running_var = torch.from_numpy(sd['base_model/' + bn_keys[i_bn] + '/moving_variance'])

    model.fc.weight.data = torch.from_numpy(sd['head_supervised/linear_layer/dense/kernel']).t()
    model.fc.weight.bias = torch.from_numpy(sd['head_supervised/linear_layer/dense/bias'])

    # 3. dump the PyTorch weights.
    torch.save({'state_dict': model.state_dict()}, output_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Model Downloader')
    parser.add_argument('--model', type=str, default='4x', choices=available_simclr_models,
                        help='The desired model')
    parser.add_argument('--simclr_category', type=str, default='pretrain',
                        choices=list(simclr_categories.keys()))
    
    args = parser.parse_args()
    model = args.model
    category = args.simclr_category
    directory = 'tf_checkpoints/' + mapping[category] + '/' + model
    input_path = directory + '/model.ckpt-' + str(simclr_categories[category][model])
    output_dir = mapping[category] 
    output_path = output_dir + '/resnet50-' + model + '.pth'
    
    os.makedirs(directory, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    run_download(model, directory, category)
    convert(input_path, output_path)


