import os
import cv2

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import nvdiffrast.torch as dr

from tqdm import tqdm
from IPython.display import Image
from torchvision.transforms import ToPILImage, ToTensor
from einops import rearrange, reduce, repeat

from diffusions.multidiffusion import MultiDiffusion
from deprecated.utils import cond_noise_sampling, erp2pers_noise_warping, compute_erp_up_noise_pred

class ERPMultiDiffusion_v3_3(MultiDiffusion):
    
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__(device, sd_version, hf_key)
        self.up_level = 3
        self.views = [
                (0.0, -22.5), (15.0, -22.5), (30.0, -22.5), (-15.0, -22.5), (-30.0, -22.5),
                (0.0,   0.0), (15.0,   0.0), (30.0,   0.0), (-15.0,   0.0), (-30.0,   0.0),
                (0.0,  22.5), (15.0,  22.5), (30.0,  22.5), (-15.0,  22.5), (-30.0,  22.5),
            ]

    @torch.no_grad()
    def text2erp(self,
                 prompts, 
                 negative_prompts='', 
                 height=512, width=1024, 
                 num_inference_steps=50,
                 guidance_scale=7.5,
                 save_dir=None):
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Define ERP source noise
        erp_latent = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)

        # Conditional white noise sampling
        erp_up_latent = cond_noise_sampling(erp_latent, self.up_level)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.no_grad():
            
            HW_pers = (64, 64)

            pers_latents, erp2pers_indices, fin_v_num =\
                erp2pers_noise_warping(erp_up_latent, HW_pers, self.views, glctx)

            imgs = []

            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                
                pers_noise_preds = []
                pers_latent_denoiseds = []

                for pers_latent in pers_latents:
                    
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    pers_latent_model_input = torch.cat([pers_latent] * 2)

                    # predict the noise residual
                    pers_noise_pred = self.unet(pers_latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                    # perform guidance
                    noise_pred_uncond, noise_pred_cond = pers_noise_pred.chunk(2)
                    pers_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    pers_noise_preds.append(pers_noise_pred)

                    # compute the denoising step with the reference model
                    pers_latent_denoised = self.scheduler.step(pers_noise_pred, t, pers_latent)['prev_sample']
                    pers_latent_denoiseds.append(pers_latent_denoised)                   

                erp_up_noise_pred = compute_erp_up_noise_pred(pers_noise_preds, erp2pers_indices, fin_v_num)

                erp_up_noise_denoised = self.scheduler.step(erp_up_noise_pred, t, erp_up_latent)['prev_sample']
                erp_up_latent = erp_up_noise_denoised

                pers_latents, _, _ = erp2pers_noise_warping(erp_up_noise_denoised, HW_pers, self.views, glctx)                               

                pers_imgs = []
                for k, pers_latent in enumerate(pers_latents):
                    pers_img = self.decode_latents(pers_latent)
                    pers_imgs.append((self.views[k], pers_img)) # [(theta, phi), img]
                imgs.append((i+1, pers_imgs)) # [i+1, [(theta, phi), img]]
                
                if save_dir is not None:
                    # save image
                    if os.path.exists(f"{save_dir}/{i+1:0>2}") is False:
                        os.mkdir(f"{save_dir}/{i+1:0>2}/")
                    for v, im in pers_imgs:
                        theta, phi = v
                        im = ToPILImage()(im[0].cpu())
                        im.save(f'/{save_dir}/{i+1:0>2}/pers_{theta}_{phi}.png')
        
        return imgs