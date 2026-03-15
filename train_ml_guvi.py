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


sw_only = False
def main(train = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    if sw_only is True:
        dataset = utils.GUVI_dataset(
            input_dir=os.path.join(fp_pd, 'guvi_paired_data', 'sw_only'),
            image_dir=os.path.join(fp_pd, 'guvi_paired_data', 'images'),
            sw_only = True
        )
        model = utils.FC_to_Conv(num_in = 4321 * 5, c_out = 1).to(device)
    
    else: 
        dataset = utils.GUVI_dataset(
            input_dir=os.path.join(fp_pd, 'guvi_paired_data', 'inputs_historical'),
            image_dir=os.path.join(fp_pd, 'guvi_paired_data', 'images')
        )
        model = utils.FC_to_Conv(num_in = 241*10, c_out = 1).to(device)
        # model = utils.FC_to_Conv(num_in = 4321*10, c_out = 1).to(device)

    # subset_indices = list(range(128*200)) 
    # dataset = Subset(dataset, subset_indices)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay = 1e-3)
    criterion = nn.HuberLoss()
    num_epochs = 3
    best_val = 100
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=True
)
    if train:
        for epoch in range(num_epochs):
            # ---------- Training ----------
            model.train()
            train_running_loss = 0.0
            val_running_loss = 0
            loop = tqdm(train_dataloader)
            for i,batch in enumerate(loop):
                x = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
                y = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)[:,:,20:]
        
                
                optimizer.zero_grad()
                y_hat = model(x)
                # y_hat = (y_hat).unsqueeze(1)
                mask = ~torch.isnan(y)
                # mask[:,:,:,48-9:48+8] = False #might need to edit this region directly with another data
                loss = criterion(y_hat[mask], y[mask])
                loss.backward()
                train_running_loss += loss.item()
                optimizer.step()
                
                loop.set_description(f"Train Loss: {loss.item():.4f}")
                
            train_loss = train_running_loss / (i+1)
            print(f'Train Loss epoch {epoch +1}: {train_loss:.4f}')
        
            
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_dataloader), 1):
                    x = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
                    y = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)[:,:,20:]
        
                    
                    y_hat = model(x)
                    # y_hat = (y_hat).unsqueeze(1)
                    mask = ~torch.isnan(y)
                    loss = criterion(y_hat[mask], y[mask])
                    val_running_loss += loss.item()
                    
                
                val_loss = val_running_loss / i 
                scheduler.step(val_loss)
    
                print(f'Val Loss epoch {epoch +1}: {val_loss:.4f}')
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), f'dummy_weight.pth')
                    print('saving new weights...')
            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
                
            
    
    
    #these generate the nice POLAR PLOTS
    
    op_mlat = np.arange(50,90, .5)
    op_mlt = np.arange(0,24,.25)
    
    model.load_state_dict(torch.load('weights/four_hour_context_0436.pth', weights_only = True))
    model.eval()
    loop = tqdm(val_dataloader)
    for batch in loop:

        x = batch['inputs'].type(torch.float32).to(device)# the hemisphere should also influence the conditioning
        y = batch['image'].type(torch.float32).unsqueeze(1)[:,:,20:]
    
        y_hat = model(x)
        
        utils.plot_auroral_grid([np.expm1(o[-1].squeeze().cpu().numpy()) for o in y],
                          op_mlat, op_mlt, title="Original Images")
        utils.plot_auroral_grid([np.expm1(p[-1].squeeze().detach().cpu().numpy()) for p in y_hat],
                          op_mlat, op_mlt, title="Predicted Images")
        
        
        # plot_auroral_grid([p[0].squeeze().cpu().numpy() for p in predictions],
        #                   op_mlat, op_mlt, title="Noisy Images")
            
    
    return train_loss_hist, val_loss_hist
if __name__ == "__main__":
    train_loss, val_loss = main(train = False)
    
    
    plt.figure()
    plt.plot(train_loss, label = 'train loss')
    plt.plot(val_loss, label = 'val loss')
    plt.legend()
    plt.show()



