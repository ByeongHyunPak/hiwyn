import os
import torch

from tqdm import tqdm
from torchvision.transforms import ToPILImage

from diffusions import ERPdiffusion_0_1_1

class HIWYN_MD_ERP2PERS_0_1_2(ERPdiffusion_0_1_1):
    """ ERPdiffusion_0_1_1 + Rotating view directions
    """
    
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__(device, sd_version, hf_key)
        self.up_level = 3
        
        self.views = [
            (0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0) ## 0.1.2 - no overlap views
        ]

        self.fin_v_nums = []
        self.indices = None
    
    @torch.no_grad()
    def encode_images(self, imgs):
        imgs = (imgs - 0.5) * 2
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents
    
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
        value = torch.zeros((1, 3, *x_erp_up.shape[-2:]), device=self.device)
        count = torch.zeros((1, 3, *x_erp_up.shape[-2:]), device=self.device)
        self.scheduler.set_timesteps(num_inference_steps)

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
                x_erp_up_j, mask_x_j = self.img_j_to_erp(img_j, j)

                value[:, :] += x_erp_up_j * mask_x_j
                count[:, :] += mask_x_j
            
            # save count map
            count_ = count / count.max()
            count_ = ToPILImage()(count_.cpu()[0][0])
            count_.save(f'/{save_dir}/{i+1:0>2}/count.png')

            # 7) Aggregate on canonical space
            x_erp_up = value / (count + 1e-8)

            ### 0.1.2 - fill the unvalid region of ERP with random color
            x_erp_up = torch.where(count!=0, x_erp_up, torch.randn((1, 3, 1, 1), device=self.device))
            ToPILImage()(x_erp_up[0].cpu()).save(f"/{save_dir}/{i+1:0>2}/erp.png")

            # 8) forward mapping x -> w^0
            ### 0.1.1 - re-encode the fused images
            # img_js = self.forward_mapping(x_erp_up)
            ### 0.1.2 - rotating view directions
            self.views = [((theta+10)%360, phi) for theta, phi in self.views]  
            img_js = self.erp_to_img_j(x_erp_up)
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