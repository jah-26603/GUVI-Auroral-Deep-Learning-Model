# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 12:42:19 2026

@author: JDawg
"""

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch # added
from torch import optim # added
from torch import nn

import utils




device = "cuda" if torch.cuda.is_available() else "cpu"


    
def ddpm_forward(
    unet: utils.ClassConditionalUNet,
    ddpm_schedule: dict,
    x_0: torch.Tensor,
    c: torch.Tensor,
    p_uncond: float,
    num_ts: int,
    eval = False
) -> torch.Tensor:
    """Algorithm 1 of the DDPM paper.

    Args:
        unet: ClassConditionalUNet
        ddpm_schedule: dict
        x_0: (N, C, H, W) input tensor.
        c: (N,) int64 condition tensor.
        p_uncond: float, probability of unconditioning the condition.
        num_ts: int, number of timesteps.

    Returns:
        (,) diffusion loss.
    """
        
    # In ddpm_forward, add this before calling unet:
    mask_uncond = torch.rand(c.shape[0]) < p_uncond
    c = c.clone()
    c[mask_uncond] = 0  # or a dedicated null token
    
    unet.train()


    t = torch.randint(0, num_ts, (x_0.shape[0],))
    eps = torch.randn_like(x_0)
    alpha_bar_t = torch.Tensor(ddpm_schedule["alpha_bars"][t]).view(-1, 1, 1, 1).to('cuda')

    mask = torch.isnan(x_0)
    x_0[mask] = 0
    xt = torch.sqrt(alpha_bar_t) * x_0.to('cuda') + torch.sqrt(1 - alpha_bar_t) * eps.to('cuda').to('cuda')
    t = t.type(torch.float).view(-1, 1).to('cuda')/num_ts

    #get a real result.

    

    output = unet(xt,c,t)
    
    # breakpoint()
    
    # snr = alpha_bar_t / (1 - alpha_bar_t)
    # # w = torch.clamp(snr, min = 1, max=10.0).to('cuda')
    # w = snr.to('cuda')
    # eps = eps.to('cuda')
    # eps*= w
    # output*=w
    criterion = nn.HuberLoss()
    loss =criterion(eps.to('cuda')[~mask],output[~mask])
    if not eval:
    # YOUR CODE HERE.
        return loss


    if eval:
        
        t = torch.randint(0, num_ts, (x_0.shape[0],))
        eps = torch.randn_like(x_0)
        alpha_bar_t = torch.Tensor(ddpm_schedule["alpha_bars"][t]).view(-1, 1, 1, 1).to('cuda')

        
        xt = torch.sqrt(alpha_bar_t) * x_0.to('cuda') + torch.sqrt(1 - alpha_bar_t) * eps.to('cuda').to('cuda')
        t = t.type(torch.float).view(-1, 1).to('cuda')/num_ts

        #get a real result.
        mask = torch.isnan(xt)
        xt[mask] = 0

        output = unet(xt,c,t)
        breakpoint()        
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(xt[0,0].detach().cpu().numpy())
        plt.title('input noisy image ')

        plt.subplot(1,3,2)
        ll = (xt - torch.sqrt(1 - alpha_bar_t) * output) / torch.sqrt(alpha_bar_t)
        plt.imshow(ll[0,0].detach().cpu().numpy())
        plt.title('new image estimate')

        plt.subplot(1,3,3)
        plt.imshow(x_0[0,0].detach().cpu().numpy())
        plt.title('ground truth')

        plt.show()
        
            
    
@torch.inference_mode()
def ddpm_sample(
    unet: utils.ClassConditionalUNet,
    ddpm_schedule: dict,
    c: torch.Tensor,
    img_wh: tuple[int, int],
    num_ts: int,
    guidance_scale: float = 5.0,
    seed: int = 0,
) -> torch.Tensor:
    """Algorithm 2 of the DDPM paper with classifier-free guidance.

    Args:
        unet: ClassConditionalUNet
        ddpm_schedule: dict
        c: (N,) int64 condition tensor. Only for class-conditional
        img_wh: (H, W) output image width and height.
        num_ts: int, number of timesteps.
        guidance_scale: float, CFG scale.
        seed: int, random seed.

    Returns:
        (N, C, H, W) final sample.
        (N, T_animation, C, H, W) caches.
    """
    unet.eval()
    
    H, W = img_wh
    C = 1 

    alpha = torch.Tensor(ddpm_schedule['alphas']).to(device)          
    alpha_bar = torch.Tensor(ddpm_schedule['alpha_bars']).to(device) 
    beta  = torch.Tensor(ddpm_schedule['betas']).to(device)       

    all_imgs = []
    for i in range(25):     
        x_t = torch.randn(C, H, W, device=device)
        xtl = []
            
        # c = torch.randint(0,9+1,(1,),).to('cuda')
        for t in reversed(range(num_ts)):
            tt = torch.Tensor([t]).type(torch.float).view(-1, 1).to('cuda')/num_ts
            x_t = x_t.squeeze()
            eps_u = unet(x_t[None,None,:,:], torch.zeros_like(c[i]).to('cuda').type(torch.float32), tt)
            eps_c = unet(x_t[None,None,:,:], c[i].type(torch.float32), tt)

            eps = eps_u + guidance_scale*(eps_c - eps_u)
            x_p0 = (alpha_bar[t])**(-1/2) * (x_t - torch.sqrt(1 - alpha_bar[t]) * eps)

            coef1 = (torch.sqrt(alpha_bar[t-1]) * beta[t]) / (1 - alpha_bar[t])
            
            # Fix: guard it
            if t > 0:
                coef1 = (torch.sqrt(alpha_bar[t-1]) * beta[t]) / (1 - alpha_bar[t])
                coef2 = (torch.sqrt(alpha[t]) * (1 - alpha_bar[t-1])) / (1 - alpha_bar[t])
                mean = coef1 * x_p0 + coef2 * x_t
                x_t = mean + torch.sqrt(beta[t]) * torch.randn_like(x_t)
            else:
                x_t = x_p0  # at t=0, just return the clean estimate directly
                
                
            xtl.append(x_t)
        all_imgs.append(xtl)
    return x_t, all_imgs


class DDPM(nn.Module):
    def __init__(
        self,
        unet: utils.ClassConditionalUNet,
        betas: tuple[float, float] = (1e-4, 0.02),
        num_ts: int = 300,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.betas = betas
        self.num_ts = num_ts
        self.p_uncond = p_uncond
        # self.ddpm_schedule = nn.ParameterDict(ddpm_schedule(betas[0], betas[1], num_ts))
        self.ddpm_schedule = ddpm_schedule(betas[0], betas[1], num_ts)


    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            c: (N,) int64 condition tensor.

        Returns:
            (,) diffusion loss.
        """
        return ddpm_forward(
            self.unet, self.ddpm_schedule, x, c, self.p_uncond, self.num_ts
        )

    @torch.inference_mode()
    def sample(
        self,
        c: torch.Tensor,
        img_wh: tuple[int, int],
        guidance_scale: float = 5.0,
        seed: int = 0,
    ):
        return ddpm_sample(
            self.unet, self.ddpm_schedule, c, img_wh, self.num_ts, guidance_scale, seed
        )


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