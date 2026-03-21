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
import torchvision.transforms as transforms





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
    


#%% Train Loop

trans = transforms.RandomErasing(value = -10)
dataset = utils.GUVI_dataset(
    input_dir=os.path.join(fp_pd, 'guvi_paired_data', 'inputs'),
    image_dir=os.path.join(fp_pd, 'guvi_paired_data', 'images'),
    transforms=trans
)

subset_indices = list(range(128*50)) 
dataset = Subset(dataset, subset_indices)
train_size = int(.70 * len(dataset))
val_size = len(dataset) - train_size


train_set, val_set = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_set,
                            batch_size=32,
                            shuffle=True,
                        )
val_dataloader = DataLoader(val_set, 
                            batch_size=32,
                            shuffle=True,
                        )

train_loss_hist, val_loss_hist= [],[]




model = utils.ClassConditionalUNet(in_channels = 1, num_classes = (240 + 1) *10 , num_hiddens = 64).to('cuda')
solver = optim.Adam(params= model.parameters(), lr = 1e-3)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = solver, gamma = .1**.05)
num_ts = 300
ddpm_sched = utils.ddpm_schedule(beta1= 1e-4, beta2= .02, num_ts = num_ts) 

pipe = utils.DDPM(unet = model)
diffusion_imgs = {}
all_loss = []
best_loss = 100

#%%
for epoch in range(5):
    loop = tqdm(train_dataloader)
    loss_list = []
    for batch in loop:
        inp = batch['inputs'].type(torch.float32)# the hemisphere should also influence the conditioning
        img = batch['image'].type(torch.float32).unsqueeze(1)
        
        loss = pipe(img,inp)
        solver.zero_grad()  
        loss.backward()
        solver.step()

        loop.set_description(f"Train Loss: {loss.item():.4f}")
        loss_list.append(loss.item())
    
    all_loss.append(np.mean(np.array(loss_list)))
    # scheduler.step()
    print(f'Epoch {epoch + 1} Loss: {all_loss[-1]:.4f}')
    if all_loss[-1] < best_loss:
        best_loss = all_loss[-1]
        torch.save(model.state_dict(), f'diffusion_weight.pth')
        print('saving new weights...')



    #after training, eval the model on multiple samples

    
plt.figure()
plt.plot(all_loss)
plt.xlabel('epoch')
plt.title('train loss vs epoch')
plt.yscale('log')
plt.show()


    
    
#%%

import matplotlib.pyplot as plt
import numpy as np

def plot_auroral_grid(images, op_mlat, op_mlt, nrows=5, ncols=5, title="Auroral Images", crop=0):
    """
    Plot a grid of auroral images in proper polar projection.
    
    Args:
        images: list of np.arrays or tensors of shape [H, W] or [C,H,W]
        op_mlat: 1D array of MLAT values corresponding to rows
        op_mlt: 1D array of MLT values corresponding to columns
        nrows, ncols: grid size
        title: figure title
        crop: int, rows to skip from the top
    """
    plt.figure(figsize=(ncols*1.6, nrows*1.6))
    
    for i in range(nrows * ncols):
        img = images[i]
        if img.ndim == 3:  # [C,H,W]
            img = img[0]
        img = img[crop:]   # optional crop
        
        # Map MLAT and MLT to polar coordinates
        mlats = op_mlat[crop:]
        r = 90 - mlats           # colatitude in degrees
        r = r / r.max()          # normalize to [0,1] for plotting
        theta = op_mlt / 24 * 2 * np.pi  # convert hours to radians
        Theta, R = np.meshgrid(theta, r)
        
        ax = plt.subplot(nrows, ncols, i+1, projection='polar')
        ax.pcolormesh(Theta, R, img, shading='auto')
        ax.set_theta_zero_location('S')  # midnight at bottom
        ax.set_theta_direction(1)        # dawn → noon → dusk
        ax.set_rlim(0, 1)                # MLAT range
        ax.set_axis_off()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

op_mlat = np.arange(40,90, .5)
op_mlt = np.arange(0,24,.25)
# Example for your datasets

# pipe = 
loop = tqdm(val_dataloader)
for batch in loop:

    inp = batch['inputs'].type(torch.float32)# the hemisphere should also influence the conditioning
    img = batch['image'].type(torch.float32).unsqueeze(1)
    plot_auroral_grid([o[-1].squeeze().cpu().numpy() for o in img],
                      op_mlat, op_mlt, title="Original Images", crop=20)
    for i in range(1):
        _, img_list = pipe.sample(c= inp, img_wh =(100,96))
        predictions = img_list 

        plot_auroral_grid([p[-1].squeeze().cpu().numpy() for p in predictions],
                          op_mlat, op_mlt, title="Predicted Images", crop=20)
    
    # plot_auroral_grid([p[0].squeeze().cpu().numpy() for p in predictions],
    #                   op_mlat, op_mlt, title="Noisy Images")






