# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:19:41 2026

@author: JDawg
"""

import matplotlib.pyplot as plt
import torch # added
from torch import optim # added
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def visualize_images_with_titles(images: torch.Tensor, column_names: list[str]):
    """
    Visualize images as a grid and title the columns with the provided names.

    Args:
        images: (N, C, H, W) tensor of images, where N is (number of rows * number of columns)
        column_names: List of column names for the titles.

    Example usage:
    visualize_images_with_titles(torch.randn(16, 1, 32, 32), ['1', '2', '3', '4'])
    """
    num_images, num_columns = images.shape[0], len(column_names)
    assert num_images % num_columns == 0, 'Number of images must be a multiple of the number of columns.'

    num_rows = num_images // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 1, num_rows * 1))

    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if i < num_columns:
            ax.set_title(column_names[i % num_columns])

    plt.tight_layout()
    plt.show()


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.Conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.GELU())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.Conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.Conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(nn.AvgPool2d(7),
                                  nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class Unflatten(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.Conv = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 7, 7, 0),
                        nn.BatchNorm2d(in_channels),
                        nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_block = nn.Sequential(Conv(in_channels, out_channels),
                                        Conv(out_channels, out_channels))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)



class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(DownConv(in_channels, out_channels),
                                        ConvBlock(out_channels, out_channels))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(UpConv(in_channels, out_channels),
                                        ConvBlock(out_channels, out_channels))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)




class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
                                    nn.Linear(in_channels, out_channels),
                                    nn.GELU(),
                                    nn.Linear(out_channels, out_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TimeConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        num_hiddens: int

    ):
        super().__init__()

        self.num_hiddens = num_hiddens

        self.x1 = ConvBlock(in_channels, num_hiddens)
        self.x2 = DownBlock(num_hiddens, num_hiddens)
        self.x3 = DownBlock(num_hiddens, num_hiddens*2)
        self.x4 = Flatten()
        self.x5 = Unflatten(num_hiddens*2)

        self.x6 = UpBlock(num_hiddens*4, num_hiddens)
        self.x7 = UpBlock(num_hiddens*2, num_hiddens)
        self.x8 = ConvBlock(num_hiddens*2, num_hiddens)
        self.x9 = nn.Conv2d(num_hiddens, out_channels, 3, 1, 1)

        self.fc1 = FCBlock(1, self.num_hiddens*2)
        self.fc2 = FCBlock(1, self.num_hiddens)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            t: (N,) normalized time tensor.

        Returns:
            (N, C, H, W) output tensor.
        """


        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)

        #differential padding for the unflattening section...
        diff_h = x3.shape[-2] - x5.shape[-2]
        diff_w = x3.shape[-1] - x5.shape[-1]
        x5 = torch.nn.functional.pad(x5, (diff_w//2, diff_w - diff_w//2, 
                                           diff_h//2, diff_h - diff_h//2))

        #these may need to be normalized, move to self
        t1 = self.fc1(t)[:,:,None,None]
        t2 = self.fc2(t)[:,:,None,None]


        x6 = torch.concatenate([x3, x5 + t1], dim = 1) 

        x7 = torch.concatenate([x2, self.x6(x6) + t2], dim = 1) 
        x8 = torch.concatenate([x1, self.x7(x7)], dim = 1)
        x9 = self.x8(x8)
        xf = self.x9(x9)

        return xf

import numpy as np



def ddpm_schedule(beta1: float, beta2: float, num_ts: int) -> dict:
    """Constants for DDPM training and sampling.

    Arguments:
        beta1: float, starting beta value.
        beta2: float, ending beta value.
        num_ts: int, number of timesteps.

    Returns:
        dict with keys:
            betas: linear schedule of betas from beta1 to beta2.
            alphas: 1 - betas.
            alpha_bars: cumulative product of alphas.
    """

    ddpm ={
        'betas': np.linspace(beta1, beta2, num = num_ts), #might need to be -1
        'alphas': 1 - np.linspace(beta1, beta2, num = num_ts) ,
        'alpha_bars': np.cumprod(1 - np.linspace(beta1, beta2, num = num_ts))
    }
    assert beta1 < beta2 < 1.0, "Expect beta1 < beta2 < 1.0."
    return ddpm



class DDPM(nn.Module):
    def __init__(
        self,
        unet: TimeConditionalUNet,
        betas: tuple[float, float] = (1e-4, 0.02),
        num_ts: int = 300,
        p_uncond: float = 0.1,
        device = 'cuda'
    ):
        super().__init__()
        self.unet = unet.to(device)
        self.num_ts = num_ts
        self.p_uncond = p_uncond

        self.ddpm_schedule = ddpm_schedule(betas[0], betas[1], num_ts)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.

        Returns:
            (,) diffusion loss.
        """



        self.unet.train()

        t = torch.randint(0, self.num_ts, (x.shape[0],))
        eps = torch.randn_like(x)
        alpha_bar_t = torch.Tensor(self.ddpm_schedule["alpha_bars"][t]).view(-1, 1, 1, 1).to('cuda')

        
        xt = torch.sqrt(alpha_bar_t) * x.to('cuda') + torch.sqrt(1 - alpha_bar_t) * eps.to('cuda').to('cuda')
        t = t.type(torch.float).view(-1, 1).to('cuda')/self.num_ts
        
        eps = eps.to('cuda')
        return eps, xt, t



    
    
    def step_back(self, x_t, eps, t):
        # Standard DDPM constants
        # (Pre-calculating these outside the loop is faster, but this works)
        alpha = torch.Tensor(self.ddpm_schedule['alphas']).to(x_t.device)          
        alpha_bar = torch.Tensor(self.ddpm_schedule['alpha_bars']).to(x_t.device) 
        beta  = torch.Tensor(self.ddpm_schedule['betas']).to(x_t.device)
    
        # 1. Predict x_0 from the noisy x_t and the predicted noise (eps)
        x_p0 = (alpha_bar[t])**(-0.5) * (x_t - torch.sqrt(1 - alpha_bar[t]) * eps)
        
        # 2. Handle t=0 edge case for alpha_bar_{t-1}
        alpha_bar_t_minus_1 = alpha_bar[t-1] if t > 0 else torch.tensor(1.0, device=x_t.device)
        
        # 3. Calculate posterior mean
        coef1 = (torch.sqrt(alpha_bar_t_minus_1) * beta[t]) / (1 - alpha_bar[t])
        coef2 = (torch.sqrt(alpha[t]) * (1 - alpha_bar_t_minus_1)) / (1 - alpha_bar[t])
        mean = coef1 * x_p0 + coef2 * x_t
    
        # 4. Add Langevin noise if not at the final step
        if t > 0:
            return mean + torch.sqrt(beta[t]) * torch.randn_like(x_t)
        else:
            return mean
    @torch.inference_mode()
    def inpaint_sample(self, original_data, mask, img_wh):
        device = 'cuda'
        batch_size = original_data.shape[0]
        
        # Start with pure noise
        x_t = torch.randn((batch_size, 2, *img_wh), device=device)
        x_t_og = x_t.clone()
        # mask = torch.abs(mask-1)
        for t in reversed(range(self.num_ts)):
            # A. Prepare the 5-channel input
            unet_input = torch.cat([x_t.to(device), mask.to(device)], dim=1)
            
            # B. Predict the noise (Corrected tt shape for batching)
            tt = torch.full((batch_size, 1), t / self.num_ts, device=device)
            eps = self.unet(unet_input, tt)
            
            # C. Step back one interval
            x_t = self.step_back(x_t, eps, t)
            
            # D. The Inpainting Trick
            if t > 0:
                noise_for_truth = torch.randn_like(original_data)
                ab = self.ddpm_schedule['alpha_bars'][t-1]
                
                # Ensure the truth is correctly noised to match the current timestep
                noisy_truth = torch.sqrt(torch.tensor(ab)) * original_data + \
                              torch.sqrt(torch.tensor(1 - ab)) * noise_for_truth
                
                # Replace 'known' pixels with noisy truth, keep 'gap' pixels as generated
                x_t = (mask.to(device) * noisy_truth.to(device)) + ((1 - mask.to(device)) * x_t)
        return x_t, x_t_og


from tqdm import tqdm
import numpy as np
from torch.utils.data import Subset
import utils
from torch.utils.data import Subset, random_split

fp_pd = r'E:\ml_aurora' #parent directory

device = 'cuda'
dataset = utils.GUVI_dataset(input_dir=os.path.join(fp_pd, 'guvi_paired_data', 'inputs'),
                              image_dir=os.path.join(fp_pd, 'guvi_paired_data', 'images'),
                                transforms = None)
subset_indices = torch.randperm(len(dataset))[:128*10]
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
impute = 0.15236633638081307
model = TimeConditionalUNet(in_channels = 4, out_channels = 2, num_classes = 1, num_hiddens = 64).to('cuda')
solver = optim.Adam(params= model.parameters(), lr = 1e-3)
num_ts = 300
ddpm_sched = ddpm_schedule(beta1= 1e-4, beta2= .02, num_ts = num_ts) 
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = solver, gamma = 1)

pipe = DDPM(unet = model)
diffusion_imgs = {}
all_loss = []
criterion = nn.MSELoss()

for epoch in range(20):
    loop = tqdm(train_dataloader)
    loss_list = []

    for batch in loop:
        
        #orderings, and masks
        hemisphere = batch['hemisphere']
        mask = batch['og_nan_mask']
        
        #operations to get image to 2d input
        batch = batch['image'].type(torch.float32).unsqueeze(1)
        empty_img = torch.full((batch.shape[0], 2, batch.shape[2], batch.shape[3]), fill_value = torch.nan)
        empty_img[torch.arange(empty_img.shape[0]), hemisphere] = batch.squeeze()
        batch = empty_img.clone()
        batch[torch.isnan(batch)] = impute
        
        mask_channel = torch.zeros_like(empty_img)
        mask_channel[torch.arange(empty_img.shape[0]), hemisphere] = 1

        #need 6 channels: 1-2 is the original image with noise added
        #predictions
        eps, xt, t = pipe(batch)
        conditioned_input = torch.cat([xt, mask_channel.to(device)], dim=1)
        
        predicted_eps = model(conditioned_input,t)
        # loss = criterion(predicted_eps, eps.to(device))
        loss = criterion(predicted_eps[mask_channel.bool()], eps[mask_channel.bool()].to(device))

        
        # preds = preds[torch.arange(preds.shape[0]), hemisphere]
        # #loss
        # eps_target  = eps[torch.arange(eps.shape[0]), hemisphere]     # ← use noise, not original image
        # loss = criterion(preds[~mask], eps_target[~mask].to(device))
        
        
        solver.zero_grad()  
        loss.backward()
        solver.step()

        loop.set_description(f"Train Loss: {loss.item():.4f}")
        loss_list.append(loss.item())
        all_loss.append(loss.item())





    print(f'Epoch {epoch + 1} loss: ', round(np.mean(np.array(loss_list)),2))
    # all_loss.append(np.mean(np.array(loss_list)))
    scheduler.step()


    if (epoch == 3 - 1) | (epoch == 20 - 1):
        xt_list, xt_og = pipe.inpaint_sample(batch, mask_channel,img_wh =(100,96))
        diffusion_imgs[f'{epoch + 1}'] = xt_list


        predictions = xt_list
        for h in range(2):
            if h == 0:
                hemi ='north'
            else:
                hemi = 'south'
                
                
            plt.figure(figsize=(8,8))
            for i in range(25):
                pred_img = xt_og[i,h].detach().cpu().numpy()
            
                plt.subplot(5,5,i+1)
                plt.imshow(pred_img)
                plt.axis('off')
            
            plt.suptitle(f"OG Images for {hemi}")
            plt.tight_layout()
            plt.show()
            
            
            plt.figure(figsize=(8,8))
            for i in range(25):
                pred_img = batch[i,h]
            
                plt.subplot(5,5,i+1)
                plt.imshow(pred_img)
                plt.axis('off')
            
            plt.suptitle(f"GT Images  for {hemi}")
            plt.tight_layout()
            plt.show()
            
            
        
            plt.figure(figsize=(8,8))
            for i in range(25):
                pred_img = predictions[i,h].detach().cpu().numpy()
            
                plt.subplot(5,5,i+1)
                plt.imshow(pred_img)
                plt.axis('off')
            
            plt.suptitle(f"Predicted Images  for {hemi}")
            plt.tight_layout()
            plt.show()
        


