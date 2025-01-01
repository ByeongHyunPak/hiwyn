import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from torchvision.transforms import ToPILImage
from diffusers import DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline

from diffusions import ERPDiffusion_0_1_1, MODEL_TYPE_STABLE_DIFFUSION, MODEL_TYPE_DEEPFLOYD
from geometry import make_coord, gridy2x_erp2pers, gridy2x_pers2erp


class DualERPDiffusion_0_0_0(ERPDiffusion_0_1_1):

    def __init__(self,
                 device,
                 hf_key=None,
                 half_precision=False,
                 fov=90,
                 views=[(0, 0)]):
        super().__init__(device, hf_key, half_precision)

        self.fov = fov
        self.views = views
        self.up_level = 3
    
    @torch.no_grad()
    def text2erp(self,
                 prompts,
                 negative_prompts='',
                 height=512, width=1024,
                 num_inference_steps=50,
                 guidance_scale=7.5, # 7.0 in visual_anagrams for DeepFloyd-IF
                 save_dir="imgs/",
                 circular_padding=True,
                 spot_diffusion=True,
                 mu=0.5):

        self.save_dir = save_dir
        
        # setting dimensions
        self.channel = self.unet.in_channels
        H = height//self.resolution_factor # ERP noise h
        W = width//self.resolution_factor # ERP noise w
        h, w = 64, 64 # Pers. noise hw (= unet input hw)

        # Check if resolution is valid for Stable Diffusion
        if self.mode == MODEL_TYPE_STABLE_DIFFUSION:
            assert (height >= 512) & (width >= 512), \
                ValueError(f"Stable Diffusion's output must have over 512 pixels. Now ({width, height}).")

        # Get text_embeds and initial noises
        text_embeds = self.prepare_text_embeds(prompts, negative_prompts)
        zt, wts = self.sample_initial_noises(H, W, h, w)

        # Get ERP branch's noise crop positions
        zt_width = W//4 + W + W//4 if circular_padding else W # ERP noise width
        zt_stride = (8, 64) if spot_diffusion else (8, 8) # stride_hw for ERP noise
        zt_views = self.get_views(H, zt_width, stride=zt_stride)

        # Define ERP fusion map
        value_z = torch.zeros((1, self.channel, H, zt_width), device=zt.device)
        count_z = torch.zeros((1, 1, H, zt_width), device=zt.device)

        # Define Pers. fusion map
        up_factor = 2 ** self.up_level
        value_w = torch.zeros((1, 3, up_factor*H, up_factor*W), device=zt.device)
        count_w = torch.zeros((1, 1, up_factor*H, up_factor*W), device=zt.device)

        # Prepare ERP-Pers. projection 
        erp_img_hw = (up_factor*H, up_factor*W)
        pers_img_hw = (64*self.resolution_factor, 64*self.resolution_factor)
        self.prepare_erp_pers_matching(erp_img_hw, pers_img_hw)

        # set scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):

            os.makedirs(f"{save_dir}/{i+1:0>2}/erp/", exist_ok=True)
            os.makedirs(f"{save_dir}/{i+1:0>2}/pers/", exist_ok=True)

            value_z.zero_(); count_z.zero_()
            value_w.zero_(); count_w.zero_()

            # Get each branch's ddim output
            zt_ddim_output = self.erp_denoising_branch(zt, text_embeds, guidance_scale, t, zt_views, value_z, count_z, spot_diffusion, circular_padding)
            wts_ddim_outputs = self.pers_denoising_branch(wts, text_embeds, guidance_scale, t)
            
            # Tweedie's formula: z0|t & w0|t
            zt_original = zt_ddim_output['pred_original_sample']
            wts_original = [wj['pred_original_sample'] for wj in wts_ddim_outputs]

            zt_original_img = self.decode_latents(zt_original)
            wts_original_img = [self.decode_latents(wj) for wj in wts_original]
            for j, w0_img in enumerate(wts_original_img):
                theta, phi = self.views[j]
                ToPILImage()(w0_img[0].cpu()).save(f"{save_dir}/{i+1:0>2}/pers/w0_{theta}_{phi}.png")

            # Upscale ERP original image by up_factor
            zt_erp_img = F.interpolate(zt_original_img, scale_factor=up_factor, mode='bilinear')
            ToPILImage()(zt_erp_img[0].cpu()).save(f"{save_dir}/{i+1:0>2}/erp/z0.png")

            # Aggregate each Pers. original image on ERP grid
            wts_erp_img = self.aggregate_pers_imgs_on_erp(wts_original_img, value_w, count_w)
            ToPILImage()(wts_erp_img[0].cpu()).save(f"{save_dir}/{i+1:0>2}/pers/w0.png")

            # Fuse zt_erp_img & wts_erp_img
            fused_erp_img = mu * zt_erp_img + (1 - mu) * wts_erp_img
            ToPILImage()(fused_erp_img[0].cpu()).save(f"{save_dir}/{i+1:0>2}/fused_erp_img.png")

            # Update zt w/ fused erp_img
            zt_original_img = F.interpolate(fused_erp_img, scale_factor=1/up_factor, mode='bilinear')
            zt_original = self.encode_images(zt_original_img)
            zt = self.get_updated_noise(zt, zt_original, t)

            # Update wt w/ fused erp_img
            wts_original_img = self.erp_to_img_j(fused_erp_img, self.pers2erp_grids)
            wts_original = [self.encode_images(wj) for wj in wts_original_img]
            wts = [self.get_updated_noise(wt, w0, t) for wt, w0 in zip(wts, wts_original)]

            # save zt &  wt images
            zt_img = self.decode_latents(zt)
            ToPILImage()(zt_img[0].cpu()).save(f"{save_dir}/{i+1:0>2}/erp/zt.png")

            wts_img = []
            for j, wt in enumerate(wts):
                theta, phi = self.views[j]
                wt_img = self.decode_latents(wt)
                ToPILImage()(wt_img[0].cpu()).save(f"{save_dir}/{i+1:0>2}/pers/wt_{theta}_{phi}.png")
                wts_img.append(wt_img)
        
        return zt_img, wts_img
        
    @torch.no_grad()
    def encode_images(self, imgs):
        imgs = (imgs - 0.5) * 2
        if self.mode == MODEL_TYPE_STABLE_DIFFUSION:
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.sample() * 0.18215
        elif self.mode == MODEL_TYPE_DEEPFLOYD:
            latents = imgs
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents):
        if self.mode == MODEL_TYPE_STABLE_DIFFUSION:
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
        elif self.mode == MODEL_TYPE_DEEPFLOYD:
            imgs = latents
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def get_updated_noise(self, xt, x0, t):
        xt_coeff, x0_coeff = self.get_xt_x0_coeff(t)
        xt_new =  xt_coeff * xt + x0_coeff * x0
        return xt_new
    
    def get_xt_x0_coeff(self, t):
        prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        std_dev_t = 0.0
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
        xt_coeff = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) / torch.sqrt(1 - alpha_prod_t)
        x0_coeff = torch.sqrt(alpha_prod_t_prev) - torch.sqrt(alpha_prod_t) / torch.sqrt(1 - alpha_prod_t) * torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2)
        return xt_coeff, x0_coeff

    @torch.no_grad()
    def aggregate_pers_imgs_on_erp(self,
                                   wts_original_img,
                                   value_w,
                                   count_w):
        
        B, C, H, W = value_w.shape

        for j, wt_img in enumerate(wts_original_img):
            per2erp_pixel_value, pers2erp_pixel_index = self.img_j_to_erp(wt_img, self.erp2pers_pairs[j])
            value_w = value_w.view(B*C, -1)
            count_w = count_w.view(B*1, -1)

            value_w[:, pers2erp_pixel_index] += per2erp_pixel_value
            count_w[:, pers2erp_pixel_index] += 1
            
        aggregated_wt_erp_img = value_w / (count_w + 1e-8)
        aggregated_wt_erp_img = aggregated_wt_erp_img.view(B, C, H, W)

        ### count map save
        count_w = count_w.view(B, 1, H, W)
        count_w = count_w / count_w.max()
        count_w = ToPILImage()(count_w.cpu()[0][0])
        count_w.save(f"{self.save_dir}/pers_branch_cnt.png")

        return aggregated_wt_erp_img

    @torch.no_grad()
    def pers_denoising_branch(self,
                              wts,
                              text_embeds,
                              guidance_scale,
                              t):
        
        wts_ddim_outputs = []

        for j, wt in enumerate(wts):

            ### classifier-free guidance
            wt_model_input = torch.cat([wt] * 2)
            wt_model_input = self.scheduler.scale_model_input(wt_model_input)

            ### unet noise prediction
            noise_pred = self.unet(wt_model_input, t, encoder_hidden_states=text_embeds)['sample']

            ### perform guidance
            noise_pred = self.noise_guidance(noise_pred, guidance_scale)

            wt_ddim_output = self.scheduler.step(noise_pred, t, wt)
            wts_ddim_outputs.append(wt_ddim_output)

        return wts_ddim_outputs


    @torch.no_grad()
    def erp_denoising_branch(self, 
                             zt,
                             text_embeds,
                             guidance_scale,
                             t,
                             zt_views,
                             value_z,
                             count_z,
                             spot_diffusion,
                             circular_padding):

        W = zt.shape[-1]

        ### horizontal translation (SpotDiffusion)
        shift_w = random.randint(0, W-1) if spot_diffusion else 0
        zt = torch.roll(zt, shifts=shift_w, dims=-1) 

        ### circular padding
        pad_w = W//4 if circular_padding else 0
        zt_padded = F.pad(zt, (pad_w, pad_w, 0, 0), mode='circular') 
        
        for h_start, h_end, w_start, w_end in zt_views:

            zt_view = zt_padded[:, :, h_start:h_end, w_start:w_end]

            ### classifier-free guidance
            zt_model_input = torch.cat([zt_view] * 2)
            zt_model_input = self.scheduler.scale_model_input(zt_model_input)

            ### unet noise prediction
            noise_pred = self.unet(zt_model_input, t, encoder_hidden_states=text_embeds)['sample']

            ### perform guidance
            noise_pred = self.noise_guidance(noise_pred, guidance_scale)

            ### save local noise_pred of zt
            value_z[:, :, h_start:h_end, w_start:w_end] += noise_pred
            count_z[:, :, h_start:h_end, w_start:w_end] += 1
        
        ### crop circular padding
        value_z = value_z[:, :, :, pad_w:-pad_w]
        count_z = count_z[:, :, :, pad_w:-pad_w]
        zt_noise_pred = value_z / (count_z + 1e-8)

        ### unroll horizontal translation (SpotDiffusion)
        zt_noise_pred = torch.roll(zt_noise_pred, shifts=-shift_w, dims=-1)
        zt = torch.roll(zt, shifts=-shift_w, dims=-1)
        zt_ddim_output = self.scheduler.step(zt_noise_pred, t, zt)
        
        ### count map save
        count_z = count_z / count_z.max()
        count_z = ToPILImage()(count_z.cpu()[0][0])
        count_z.save(f"{self.save_dir}/erp_branch_cnt.png")

        return zt_ddim_output


    # misc
    @torch.no_grad()
    def prepare_erp_pers_matching(self, erp_img_HW, pers_img_HW):
        '''
        erp2pers_pairs: list of (erp2pers_grid, erp_indices)
        '''

        erp2pers_pairs = [] 
        pers2erp_grids = []
        
        erp_grid = make_coord(erp_img_HW, flatten=False).to(self.device) # (H, W, 2)
        pers_grid = make_coord(pers_img_HW, flatten=False).to(self.device) # (h, w, 2)

        for theta, phi in self.views:  
            ### ERP2PERS ###      
            erp2pers_grid, valid_mask = gridy2x_erp2pers(gridy=erp_grid,
                HWy=erp_img_HW, HWx=pers_img_HW, THETA=theta, PHI=phi, FOVy=360, FOVx=90) # (H*W, 2), (H*W)
        
            # Filter valid indices
            erp_indices = torch.arange(0, erp_img_HW[0]*erp_img_HW[1]).to(self.device) # (H*W,)
            erp_indices = erp_indices[valid_mask.bool()].long() # sample (D,) from (H*W,)

            erp2pers_grid = erp2pers_grid[valid_mask.bool()] # sample (D, 2) from (H*W, 2)
            erp2pers_grid = erp2pers_grid.unsqueeze(1).unsqueeze(0) # (D, 2) -> (1, D, 1, 2)
            erp2pers_grid = erp2pers_grid.flip(-1) # x <-> y

            erp2pers_pairs.append((erp2pers_grid, erp_indices))

            ### PERS2ERP ###
            pers2erp_grid, valid_mask = gridy2x_pers2erp(gridy=pers_grid,
                HWy=pers_img_HW, HWx=erp_img_HW, THETA=theta, PHI=phi, FOVy=self.fov, FOVx=360)
            pers2erp_grid = pers2erp_grid.view(*pers_img_HW, 2).unsqueeze(0).flip(-1)
            pers2erp_grids.append(pers2erp_grid)

        self.erp2pers_pairs = erp2pers_pairs
        self.pers2erp_grids = pers2erp_grids
        return erp2pers_pairs, pers2erp_grids

    @torch.no_grad()
    def noise_guidance(self, noise_pred, guidance_scale):

        if self.mode == MODEL_TYPE_STABLE_DIFFUSION:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        elif self.mode == MODEL_TYPE_DEEPFLOYD:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(self.channel, dim=1)
            noise_pred_cond, predicted_variance = noise_pred_cond.split(self.channel, dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
            noise_pred, _ = noise_pred.split(self.channel, dim=1)
        
        return noise_pred
    
    @torch.no_grad()
    def prepare_text_embeds(self, prompts, negative_prompts):
        prompts = [prompts] if isinstance(prompts, str) else prompts
        negative_prompts = [negative_prompts] if isinstance(negative_prompts, str) else negative_prompts
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]
        return text_embeds

    @torch.no_grad()
    def get_views(self, panorama_height, panorama_width, window_size=64, stride=(8, 8)):
        if isinstance(stride, int):
            stride = (stride, stride)
        stride_h, stride_w = stride
        panorama_height /= self.resolution_factor
        panorama_width /= self.resolution_factor
        num_blocks_height = (panorama_height - window_size) // stride_h + 1
        num_blocks_width = (panorama_width - window_size) // stride_w + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride_h)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride_w)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
        return views

    @torch.no_grad()
    def cond_noise_upsampling(self, src_noise):

        B, C, H, W = src_noise.shape
        up_factor = 2 ** self.up_level
        upscaled_means = F.interpolate(src_noise, scale_factor=(up_factor, up_factor), mode='nearest')

        up_H = up_factor * H
        up_W = up_factor * W

        # 1) Unconditionally sample a discrete Nk x Nk Gaussian sample
        raw_rand = torch.randn(B, C, up_H, up_W, device=src_noise.device)

        # 2) Remove its mean from it
        Z_mean = raw_rand.unfold(2, up_factor, up_factor).unfold(3, up_factor, up_factor).mean((4, 5))
        Z_mean = F.interpolate(Z_mean, scale_factor=up_factor, mode='nearest')
        mean_removed_rand = raw_rand - Z_mean

        # 3) Add the pixel value to it
        up_noise = upscaled_means / up_factor + mean_removed_rand

        return up_noise

    @torch.no_grad()
    def erp2pers_discrete_warping(self, erp_noise, pers_hw):
        
        B, C, H, W = erp_noise.shape
        h, w = pers_hw

        erp_noise_flat = erp_noise.reshape(B*C, -1) # (B*C, H*W)
        erp_pixel_grid = make_coord((H, W), flatten=True).to(self.device) # (H*W, 2)
        pers_pixel_idx = torch.arange(1, h*w+1, dtype=torch.float32).to(self.device) # (h*w, )
        count_values = torch.ones_like(erp_noise_flat[:1]) # (1, H*W)

        pers_noises = []

        for theta, phi in self.views:
            # 1) Map each ERP pixel on Pers. image grid
            erp2pers_grid, valid_mask = gridy2x_erp2pers(gridy=erp_pixel_grid,
                HWy=(H, W), HWx=(h, w), THETA=theta, PHI=phi, FOVy=360, FOVx=self.fov)
            valid_erp2pers_grid = erp2pers_grid[valid_mask.bool()] # (D, 2) for D < H*W
            
            # 2) Find nearest Pers. pixel index of each erp pixel
            valid_erp2pers_idx = F.grid_sample(
                pers_pixel_idx.view(1, 1, h, w),
                valid_erp2pers_grid.view(1, 1, -1, 2).flip(-1),
                mode="nearest", align_corners=False).view(1, -1) # (1, D)
            valid_erp2pers_idx = valid_erp2pers_idx.to(torch.int64)

            # 3) Get warped Pers. noise
            fin_v_val = torch.zeros(B*C, h*w+1, device=self.device).scatter_add_(
                1, index=valid_erp2pers_idx.repeat(B*C, 1), src=erp_noise_flat)[..., 1:] # (B*C, h*w)
            fin_v_num = torch.zeros(1, h*w+1, device=self.device).scatter_add_(
                1, index=valid_erp2pers_idx, src=count_values)[..., 1:] # (1, h*w)
            
            assert fin_v_num.min() != 0, ValueError(f"There are some pixels that do not match with any ERP pixel. ({theta},{phi})")
            final_values = fin_v_val / torch.sqrt(fin_v_num)
            pers_noise = final_values.reshape(B, C, h, w)
            pers_noise = pers_noise.to(self.device)
            pers_noises.append(pers_noise)
        
        return pers_noises

    def sample_initial_noises(self, H, W, h, w):
        zt = torch.randn((1, self.channel, H, W), device=self.device) # 1) Sample initial zT (ERP noise)
        zt_up = self.cond_noise_upsampling(zt) # 2) Conditional Upsampling zT
        wts = self.erp2pers_discrete_warping(zt_up, (h, w)) # 3) Get wTs from zT_up by discrete warping
        return zt, wts