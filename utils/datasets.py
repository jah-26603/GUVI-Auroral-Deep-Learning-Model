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
import torchvision.transforms as transforms


#%%

class GUVI_dataset(Dataset):
    '''GUVI dataset'''
    def __init__(self, 
                 input_dir =r'E:\ml_aurora\guvi_paired_data\inputs',
                 image_dir = r'E:\ml_aurora\guvi_paired_data\images',
                 transforms = None,
                 sw_only = False):

        self.input_dir = input_dir
        self.image_dir = image_dir
        self.titles = os.listdir(input_dir)
        
        with open("config.yaml") as stream:
            stat_dict =yaml.safe_load(stream)
        
        #values determined across entire dataset
        self.input_mean = np.array(stat_dict['data_statistics']['input_mean'])
        self.input_std = np.array(stat_dict['data_statistics']['input_std'])
        self.image_mean = stat_dict['data_statistics']['all_guvi_img_mean']
        self.image_std = stat_dict['data_statistics']['all_guvi_img_std']
        # self.image_max = stat_dict['data_statistics']['guvi_img_max']
        self.transforms = transforms
        
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
        img = np.load(os.path.join(self.image_dir, self.titles[idx]))[20:]
        # img = (np.log(img) - self.image_mean)/ self.image_std
        # img[img == -np.inf] = np.nan
        img = (img - self.image_mean)/ self.image_std
        

        
        if 'north' in self.titles[idx]:
            hemisphere = 0
        if 'south' in self.titles[idx]:
            hemisphere = 1
        
        
        mask = np.isnan(img)
        sample = {'image': img, 'inputs': inputs, 'hemisphere': hemisphere, 'og_nan_mask': mask}

        return sample


   

