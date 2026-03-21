# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 23:23:43 2025

@author: JDawg
"""

import utils
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pytorch_msssim import ssim
import glob
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import os 
from torch.utils.data import Subset
from torch import optim # added





device = "cuda" if torch.cuda.is_available() else "cpu"
initial_run = False

fp_pd = r'E:\ml_aurora' #parent directory

#will download all necessary training data and pair
if initial_run:
    utils.download_solar_wind_data(
        out_dir = os.path.join(fp_pd, 'solar_wind')
        ) #downloads files
    
    utils.collate_solar_wind(
        fp = os.path.join(fp_pd, 'solar_wind'), 
        out = os.path.join(fp_pd, 'organized_solar_wind.csv')
        ) #creates the solar wind dataframe  
    
    utils.guvi_to_images(
        og_data_fp=os.path.join(fp_pd, 'guvi_aurora'),
        out=os.path.join(fp_pd, 'guvi_paired_data', 'images')
    )
    
    utils.guvi_input_data(
        guvi_images_fp=os.path.join(fp_pd, 'guvi_paired_data', 'images'),
        guvi_inputs_fp=os.path.join(fp_pd, 'guvi_paired_data', 'inputs')
    )
    # utils.guvi_input_data(
    #     guvi_images_fp=os.path.join(fp_pd, 'guvi_paired_data', 'images'),
    #     guvi_inputs_fp=os.path.join(fp_pd, 'guvi_paired_data', 'inputs_historical'),
    #     historical = True
    # )

# utils.guvi_input_data(
#     guvi_images_fp=os.path.join(fp_pd, 'guvi_paired_data', 'images'),
#     guvi_inputs_fp=os.path.join(fp_pd, 'guvi_paired_data', 'three_day_forecast'),
#     historical = False,
#     sw_only = True
# )
            
#the historical is technically incorrect as it is taking inputs from the current solar wind
# which is 1 hour ahead, breaking continuity. Need solar wind from 5 to 1 hour ago and current indices


    
#%%


#%%

train = True
sw_only = False
# def main(train = True):
device = "cuda" if torch.cuda.is_available() else "cpu"


if sw_only is True:
    dataset = utils.GUVI_dataset(
        input_dir=os.path.join(fp_pd, 'guvi_paired_data', 'sw_only'),
        image_dir=os.path.join(fp_pd, 'guvi_paired_data', 'images'),
        sw_only = True
    )
    model = utils.FC_to_Conv(num_in = 4320 * 5, c_out = 2).to(device)

else: 
    dataset = utils.GUVI_dataset(
        input_dir=os.path.join(fp_pd, 'guvi_paired_data', 'inputs'),
        image_dir=os.path.join(fp_pd, 'guvi_paired_data', 'images')
    )
    model = utils.FC_to_Conv(num_in = 240*10, c_out = 2).to(device)
    # model = utils.FC_to_Conv(num_in = 4321*10, c_out = 1).to(device)

subset_indices = torch.randperm(len(dataset))[:128*1]
dataset = Subset(dataset, subset_indices)
train_size = int(.70 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator()
generator.manual_seed(0)

train_set, val_set = random_split(dataset, [train_size, val_size], generator = generator)
train_dataloader = DataLoader(train_set,
                            batch_size=32,
                            shuffle=True,
                        )
val_dataloader = DataLoader(val_set, 
                            batch_size=32,
                            shuffle=True,
                        )

train_loss_hist, val_loss_hist= [],[]


# Better optimizer settings
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 1e-3)
criterion = nn.HuberLoss()
num_epochs = 5
best_val = 100

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
# optimizer,
# mode="min",
# factor=0.5,
# patience=2,
# min_lr=1e-6,
# verbose=True
# )

imgs_std = 1.72481
imgs_mean = 0.475354
# impute = -imgs_mean/imgs_std
# impute = 0


if train:
    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        train_running_loss = 0.0
        val_running_loss = 0
        loop = tqdm(train_dataloader)
        for i,batch in enumerate(loop):
            
            
            inputs = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
            images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
            hemisphere = batch["hemisphere"].to(device, dtype=torch.long)
            nan_mask = batch["og_nan_mask"].unsqueeze(1).to(device, dtype=torch.float32) #1's = nan, 0's = data


            B, _, H, W = images.shape
            empty_img = torch.full((B, 2, H, W), torch.nan, device=device)
            mask_channel = torch.zeros_like(empty_img).to(device).int()
            for i in range(B):
                empty_img[i, hemisphere[i]] = images[i, 0]
                mask_channel[i, hemisphere[i]] = nan_mask[i,0].int()

            images = empty_img.clone()
            # images[torch.isnan(images)] = impute
            
            
            optimizer.zero_grad()
            y_hat = model(inputs)
            mask_channel[i, 1 - hemisphere[i]] = 1
            mask = ~mask_channel.bool()
            mask = ~torch.isnan(images)
            loss = criterion(y_hat[mask], images[mask])
            
            # breakpoint()
            loss.backward()
            train_running_loss += loss.item()
            optimizer.step()
            
            loop.set_description(f"Train Loss: {loss.item():.4f}")
            
        train_loss = train_running_loss / (i+1)
        print(f'Train Loss epoch {epoch +1}: {train_loss:.4f}')
    
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader), 1):

    
    
                inputs = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
                images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
                hemisphere = batch["hemisphere"].unsqueeze(1).to(device, dtype=torch.int)
                nan_mask = batch["og_nan_mask"].unsqueeze(1).to(device, dtype=torch.float32) #1's = nan, 0's = data
    
    
                B, _, H, W = images.shape
                empty_img = torch.full((B, 2, H, W), torch.nan, device=device)
                mask_channel = torch.zeros_like(empty_img).to(device).int()
                for i in range(B):
                    empty_img[i, hemisphere[i]] = images[i, 0]
                    mask_channel[i, hemisphere[i]] = nan_mask[i,0].int()
    
                images = empty_img.clone()
                # images[torch.isnan(images)] = impute
    
                
                y_hat = model(inputs)
                mask = ~torch.isnan(images)
                loss = criterion(y_hat[mask], images[mask])
                
                val_running_loss += loss.item()
                
            
            val_loss = val_running_loss / i 
            # scheduler.step(val_loss)

            print(f'Val Loss epoch {epoch +1}: {val_loss:.4f}')
            # if val_loss < best_val:
            #     best_val = val_loss
            #     torch.save(model.state_dict(), f'dummy_weight.pth')
            #     print('saving new weights...')
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
            
        


#these generate the nice POLAR PLOTS

op_mlat = np.arange(50,90, .5)
op_mlt = np.arange(0,24,.25)

# model.load_state_dict(torch.load('weights/four_hour_context_0436.pth', weights_only = True))
model.eval()
loop = tqdm(train_dataloader)
for batch in loop:

    inputs = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
    images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
    hemisphere = batch["hemisphere"].unsqueeze(1).to(device, dtype=torch.int)
    nan_mask = batch["og_nan_mask"].unsqueeze(1).to(device, dtype=torch.float32) #1's = nan, 0's = data


    B, _, H, W = images.shape
    empty_img = torch.full((B, 2, H, W), torch.nan, device=device)
    mask_channel = torch.zeros_like(empty_img).to(device).int()
    for i in range(B):
        empty_img[i, hemisphere[i]] = images[i, 0]
        mask_channel[i, hemisphere[i]] = nan_mask[i,0].int()

    images = empty_img.clone()
    # images[torch.isnan(images)] = impute

    y_hat = model(inputs)
    
    utils.plot_auroral_grid([o[-1].squeeze().cpu().numpy()*imgs_std + imgs_mean for o in images],
                      op_mlat, op_mlt, title="Original Images")
    utils.plot_auroral_grid([p[-1].squeeze().detach().cpu().numpy() *imgs_std + imgs_mean for p in y_hat],
                      op_mlat, op_mlt, title="Predicted Images")
    
    
    # plot_auroral_grid([p[0].squeeze().cpu().numpy() for p in predictions],
    #                   op_mlat, op_mlt, title="Noisy Images")
        

    # return train_loss_hist, val_loss_hist
# if __name__ == "__main__":
#     train_loss, val_loss = main(train = True)
    
    
#     plt.figure()
#     plt.plot(train_loss, label = 'train loss')
#     plt.plot(val_loss, label = 'val loss')
#     plt.legend()
#     plt.show()



