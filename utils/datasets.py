# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 10:43:33 2025

@author: JDawg
"""

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import yaml
import matplotlib.pyplot as plt
import cv2


#%%

class GUVI_dataset(Dataset):
    '''GUVI dataset'''
    def __init__(self, 
                 input_dir =r'E:\ml_aurora\guvi_paired_data\inputs',
                 image_dir = r'E:\ml_aurora\guvi_paired_data\images',
                 sw_only = False):

        self.input_dir = input_dir
        self.image_dir = image_dir
        self.titles = os.listdir(input_dir)
        
        with open("config.yaml") as stream:
            stat_dict =yaml.safe_load(stream)
        
        #values determined across entire dataset
        self.input_mean = np.array(stat_dict['data_statistics']['input_mean'])
        self.input_std = np.array(stat_dict['data_statistics']['input_std'])
        self.image_mean = np.array(stat_dict['data_statistics']['guvi_img_mean'])
        self.image_std = np.array(stat_dict['data_statistics']['guvi_img_std'])
        
        if sw_only:
            self.input_mean = self.input_mean[:5]
            self.input_std = self.input_std[:5]
            
    def __len__(self):
        return len(os.listdir(self.input_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
            
        #load and standardize the inputs
        inputs = np.load(os.path.join(self.input_dir, self.titles[idx])).astype(float)
        inputs = (inputs - self.input_mean)/self.input_std
        mask = np.isnan(inputs)
        inputs[mask] = 0 #the outputs will be masked during training
        
        #load and standardize the images, and add hemisphere conditioning to the inputs
        img = np.load(os.path.join(self.image_dir, self.titles[idx]))
        if 'south' in self.titles[idx]:
            # img = (img - self.image_mean[1])/ self.image_std[1]
            inputs = np.concatenate([inputs, np.array([1]*inputs.shape[1])[None,:]], axis = 0)
        if 'north' in self.titles[idx]:
            # img = (img - self.image_mean[0])/ self.image_std[0]
            inputs = np.concatenate([inputs, np.array([0]*inputs.shape[1])[None,:]], axis = 0)

        # #salt and pepper noise removal? -> this could be hurting the noon sector
        # mask = np.isnan(img)
        # img[mask] = 0
        # img = cv2.medianBlur(img.astype(np.float32), ksize=3)
        # img[mask] = np.nan
        
        img = np.log1p(img)

        sample = {'image': img, 'inputs': inputs}

        return sample


   

