import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torchvision.transforms import ToPILImage

from diffusions import ERPDiffusion_0_1_0
from geometry import make_coord, gridy2x_erp2pers, gridy2x_pers2erp

class ERPDiffusion_0_1_1(ERPDiffusion_0_1_0):
    """ HIWYN + SyncTweedies (fusion on decoder(w_0), not w_0(=ERPDiffusion_0_1_0))
    """
    
    def __init__(self, 
                 device, 
                 hf_key=None, 
                 fov=90,
                 views=[(0, 0), (45, 0)],
                 half_precision=False):
        super().__init__(device, hf_key, half_precision)

        self.up_level = 3
        
        self.views = views
        self.fov = fov

        self.fin_v_nums = []
        self.indices = None
    
    @torch.no_grad()
    def encode_images(self, imgs):
        imgs = (imgs - 0.5) * 2
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents
    
    @torch.no_grad()
    def img_j_to_erp(self, img_j, erp2pers_pair, erp_HW=(1024, 2048)):
        H, W = erp_HW

        erp2pers_grid, erp_indices = erp2pers_pair # (_, D, _, 2), (D,)

        img_j_to_erp_img = F.grid_sample(
            img_j, # (1, C, h, w)
            erp2pers_grid, # (_, D, _, 2)
            mode="bicubic", align_corners=False,
            padding_mode="reflection"
        ) # (_, C, D, 1)

        img_j_to_erp_img.clamp_(0, 1)

        img_j_to_erp_img = img_j_to_erp_img.view(img_j.shape[1], -1) # (_, C, D, 1) -> (C, D)

        return img_j_to_erp_img, erp_indices # (C, D), (D,); value, indices
    
    @torch.no_grad()
    def erp_to_img_j(self, erp_img, pers2erp_grids):
        B, C, H, W = erp_img.shape

        pers2erp_grids_input = torch.stack(pers2erp_grids).squeeze(1) 

        img_js = F.grid_sample(
            erp_img.expand(len(self.views), -1, -1 ,-1),
            pers2erp_grids_input,
            mode="bicubic", align_corners=False)
        
        img_js.clamp_(0, 1)
        
        img_js = img_js.unsqueeze(1)
        
        return list(torch.unbind(img_js, dim=0))

    '''
    Prepare each view's Perspective coordinates w.r.t. ERP indices (erp2pers_grid) and corresponding ERP indices (erp_indices)
    '''
    @torch.no_grad()
    def prepare_erp_pers_matching(self, erpHW, imgHW=(512, 512)):
        '''
        erp2pers_pairs: list of (erp2pers_grid, erp_indices)
        '''

        erp2pers_pairs = [] 
        pers2erp_grids = []
        
        erp_grid = make_coord(erpHW, flatten=False).to(self.device) # (H, W, 2); ERP coordinates
        pers_grid = make_coord(imgHW, flatten=False).to(self.device) # (h, w, 2)

        for theta, phi in self.views:  
            ### ERP2PERS ###      
            erp2pers_grid, valid_mask = gridy2x_erp2pers(gridy=erp_grid,
                HWy=erpHW, HWx=imgHW, THETA=theta, PHI=phi, FOVy=360, FOVx=90) # (H*W, 2), (H*W)
        
            # Filter valid indices
            erp_indices = torch.arange(0, erpHW[0]*erpHW[1]).to(self.device) # (H*W,)
            erp_indices = erp_indices[valid_mask.bool()].long() # sample (D,) from (H*W,)

            erp2pers_grid = erp2pers_grid[valid_mask.bool()] # sample (D, 2) from (H*W, 2)
            erp2pers_grid = erp2pers_grid.unsqueeze(1).unsqueeze(0) # (D, 2) -> (1, D, 1, 2)
            erp2pers_grid = erp2pers_grid.flip(-1) # x <-> y

            erp2pers_pairs.append((erp2pers_grid, erp_indices))

            ### PERS2ERP ###
            pers2erp_grid, valid_mask = gridy2x_pers2erp(gridy=pers_grid,
                HWy=imgHW, HWx=erpHW, THETA=theta, PHI=phi, FOVy=self.fov, FOVx=360)
            pers2erp_grid = pers2erp_grid.view(*imgHW, 2).unsqueeze(0).flip(-1)
            pers2erp_grids.append(pers2erp_grid)

        return erp2pers_pairs, pers2erp_grids

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

        # 1) Initialize x^T_ERP
        x_erp = torch.randn((1, 4, height//8, width//8), device=self.device)

        # 2) Conditional Upsampling x^T_ERP
        x_erp_up = self.cond_noise_sampling(x_erp)

        # 3) Discrete warping x^T_ERP to {w^T_i}_i=1:N
        w_js = self.discrete_warping(x_erp_up)
        buffer_imgs = self.decode_and_save(-1, w_js, save_dir, [])

        # set scheduler
        # value = torch.zeros_like(x_erp_up)
        # count = torch.zeros_like(x_erp_up)
        H, W = x_erp_up.shape[-2:]
        value = torch.zeros((1, 3, H, W), device=self.device)
        count = torch.zeros((1, 3, H, W), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        erp2pers_pairs, pers2erp_grids = self.prepare_erp_pers_matching(erpHW=(H, W))        

        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            if os.path.exists(f"{save_dir}/{i+1:0>2}") is False:
                    os.mkdir(f"{save_dir}/{i+1:0>2}/")

            value.zero_()
            count.zero_()

            for j, w_j in enumerate(w_js):

                # 4) Predict e_theta(w^t_j, t)
                unet_input = torch.cat([w_j] * 2)
                noise_pred = self.unet(unet_input, t, encoder_hidden_states=text_embeds)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # 5) Get w'^t-1_j, w'^0_j
                ddim_output = self.scheduler.step(noise_pred, t, w_j)
                w_j_denoised = ddim_output['prev_sample']
                w_j_original = ddim_output['pred_original_sample']
                
                ### 0.1.1 - fusion on decoded latents
                img_j = self.decode_latents(w_j_original)               
                theta, phi = self.views[j]
                ToPILImage()(img_j[0].cpu()).save(f'/{save_dir}/{i+1:0>2}/w^0_{theta}_{phi}.png')
                # img_j = F.interpolate(img_j, scale_factor=1/8, mode='bilinear', align_corners=False)

                # 6) inverse mapping w'^0 -> x
                # x_erp_up_j, mask_x_j = self.inverse_mapping(img_j, j)
                x_erp_up_j, indices = self.img_j_to_erp(img_j, erp2pers_pairs[j])

                value_ = value.view(3, H*W)
                count_ = count.view(3, H*W)
                value_[:, indices] += x_erp_up_j
                count_[:, indices] += 1
            
            # save count map
            count_ = count / count.max()
            count_ = ToPILImage()(count_.cpu()[0][0])
            count_.save(f'/{save_dir}/{i+1:0>2}/count.png')

            # 7) Aggregate on canonical space
            x_erp_up = value / (count + 1e-8)
            ToPILImage()(x_erp_up[0].cpu()).save(f"/{save_dir}/{i+1:0>2}/erp.png")

            # save count map
            count_ = (count / count.max()).view(1, 3, *x_erp_up.shape[-2:])
            count_ = ToPILImage()(count_.cpu()[0][0])
            count_.save(f'/{save_dir}/{i+1:0>2}/count.png')

            # 7) Aggregate on canonical space
            x_erp_up = value / (count + 1e-8)
            x_erp_up = x_erp_up.view(1, 3, *x_erp_up.shape[-2:])
            ToPILImage()(x_erp_up[0].cpu()).save(f"/{save_dir}/{i+1:0>2}/erp.png")

            # 8) forward mapping x -> w^0
            ### 0.1.1 - re-encode the fused images
            # img_js = self.forward_mapping(x_erp_up)
            img_js = self.erp_to_img_j(x_erp_up, pers2erp_grids)
            w0_js = []
            for img_j in img_js:
                # img_j = F.interpolate(img_j, scale_factor=8, mode='bicubic', align_corners=False)
                w0_j = self.encode_images(img_j)
                w0_js.append(w0_j)            
            
            # 9) ddim scheduler step w/ w^t & w^0
            w_new_js = []
            for w_j, w0_j in zip(w_js, w0_js):
                timestep = t
                prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

                alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

                beta_prod_t = 1 - alpha_prod_t

                variance = self.scheduler._get_variance(timestep, prev_timestep)
                std_dev_t = 0.0

                xt_coeff = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) / torch.sqrt(1 - alpha_prod_t)
                x0_coeff = torch.sqrt(alpha_prod_t_prev) - torch.sqrt(alpha_prod_t) / torch.sqrt(1 - alpha_prod_t) * torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2)
                
                w_new_j = xt_coeff * w_j + x0_coeff * w0_j
                w_new_js.append(w_new_j)
            w_js = w_new_js
            
            buffer_imgs = self.decode_and_save(i, w_js, save_dir, buffer_imgs)
        
        return buffer_imgs