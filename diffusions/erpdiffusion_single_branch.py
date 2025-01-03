import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision.transforms import ToPILImage

from geometry import prepare_erp_pers_matching, img_j_to_erp, erp_to_img_j

class ERPDiffusionSingleBranch(nn.Module):
    """ HIWYN + SyncTweedies (fusion on decoder(w_0), not w_0(=ERPDiffusion_0_1_0))
    """
    
    def __init__(self, 
                 pipe,
                 device, 
                 fov=90,
                 views=[(0, 0), (45, 0)]):
        super().__init__()

        self.pipe = pipe

        self.device = device
        
        self.up_level = 3
        
        self.views = views
        self.fov = fov

        self.fin_v_nums = []
        self.indices = None

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

        h = w = self.pipe.unet.config.sample_size

        # Prompts -> text embeds
        text_embeds = self.pipe.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # # 1) Initialize x^T_ERP
        # x_erp = torch.randn((1, 4, height//8, width//8), device=self.device)

        # 2) Conditional Upsampling x^T_ERP and Discrete warping x^T_ERP to {w^T_i}_i=1:N
        x_erp_up, w_js = self.pipe.sample_initial_noises(self.views, self.fov, height, width, h, w)

        buffer_imgs = self.pipe.decode_and_save(-1, self.views, w_js, save_dir, [])

        # set scheduler
        # value = torch.zeros_like(x_erp_up)
        # count = torch.zeros_like(x_erp_up)
        H, W = x_erp_up.shape[-2:]
        value = torch.zeros((1, 3, H, W), device=self.device)
        count = torch.zeros((1, 3, H, W), device=self.device)

        self.pipe.scheduler.set_timesteps(num_inference_steps)

        erp2pers_pairs, pers2erp_grids = prepare_erp_pers_matching(self.views, self.fov, self.device, erpHW=(H, W))        

        for i, t in enumerate(tqdm(self.pipe.scheduler.timesteps)):
            if os.path.exists(f"{save_dir}/{i+1:0>2}") is False:
                    os.mkdir(f"{save_dir}/{i+1:0>2}/")

            value.zero_()
            count.zero_()

            for j, w_j in enumerate(w_js):

                # 4) Predict e_theta(w^t_j, t)
                unet_input = torch.cat([w_j] * 2)
                noise_pred = self.pipe.unet(unet_input, t, encoder_hidden_states=text_embeds)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # 5) Get w'^t-1_j, w'^0_j
                ddim_output = self.pipe.scheduler.step(noise_pred, t, w_j)
                w_j_denoised = ddim_output['prev_sample']
                w_j_original = ddim_output['pred_original_sample']
                
                ### 0.1.1 - fusion on decoded latents
                img_j = self.pipe.decode_latents(w_j_original)               
                theta, phi = self.views[j]
                ToPILImage()(img_j[0].cpu()).save(f'/{save_dir}/{i+1:0>2}/w^0_{theta}_{phi}.png')
                # img_j = F.interpolate(img_j, scale_factor=1/8, mode='bilinear', align_corners=False)

                # 6) inverse mapping w'^0 -> x
                # x_erp_up_j, mask_x_j = self.inverse_mapping(img_j, j)
                x_erp_up_j, indices = img_j_to_erp(img_j, erp2pers_pairs[j])

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
            img_js = erp_to_img_j(x_erp_up, pers2erp_grids)
            w0_js = []
            for img_j in img_js:
                # img_j = F.interpolate(img_j, scale_factor=8, mode='bicubic', align_corners=False)
                w0_j = self.pipe.encode_images(img_j)
                w0_js.append(w0_j)            
            
            # 9) ddim scheduler step w/ w^t & w^0
            w_new_js = []
            for w_j, w0_j in zip(w_js, w0_js):
                timestep = t
                prev_timestep = timestep - self.pipe.scheduler.config.num_train_timesteps // self.pipe.scheduler.num_inference_steps

                alpha_prod_t = self.pipe.scheduler.alphas_cumprod[timestep]
                alpha_prod_t_prev = self.pipe.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.pipe.scheduler.final_alpha_cumprod

                beta_prod_t = 1 - alpha_prod_t

                variance = self.pipe.scheduler._get_variance(timestep, prev_timestep)
                std_dev_t = 0.0

                xt_coeff = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) / torch.sqrt(1 - alpha_prod_t)
                x0_coeff = torch.sqrt(alpha_prod_t_prev) - torch.sqrt(alpha_prod_t) / torch.sqrt(1 - alpha_prod_t) * torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2)
                
                w_new_j = xt_coeff * w_j + x0_coeff * w0_j
                w_new_js.append(w_new_j)
            w_js = w_new_js
            
            buffer_imgs = self.pipe.decode_and_save(i, self.views, w_js, save_dir, buffer_imgs)
        
        return buffer_imgs