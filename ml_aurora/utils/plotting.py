# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 11:27:41 2026

@author: JDawg
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


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
    fig = plt.figure(figsize=(ncols*1.5, nrows*1.5/1.21))
    norm = Normalize(vmin=0, vmax=3)
    # norm = Normalize(vmin=0, vmax=.25)

    
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
        im = ax.pcolormesh(Theta, R, img, shading='auto',
                   norm=norm)
        
        ax.set_theta_zero_location('S')  # midnight at bottom
        ax.set_theta_direction(1)        # dawn → noon → dusk
        ax.set_rlim(0, 1)                # MLAT range
        ax.set_axis_off()
    cbar = fig.colorbar(im, ax=fig.axes)
    cbar.set_label(r'ergs cm$^{-2}$ s$^{-1}$')
    plt.suptitle(title)
    # plt.tight_layout()
    plt.show()