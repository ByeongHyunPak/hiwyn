import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torchvision.transforms import ToPILImage

from diffusions import ERPDiffusion_0_0_0

class ERPDiffusion_0_1_0(ERPDiffusion_0_0_0):
    """ HIWYN + SyncTweedies (fusion on w_0, not w_t(=MultiDiffusion))
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
        value = torch.zeros_like(x_erp_up)
        count = torch.zeros_like(x_erp_up)
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
                img_j = self.decode_latents(w_j_original)               
                theta, phi = self.views[j]
                ToPILImage()(img_j[0].cpu()).save(f'/{save_dir}/{i+1:0>2}/w^0_{theta}_{phi}.png')
                
                # 6) inverse mapping w'^0 -> x
                x_erp_up_j, mask_x_j = self.inverse_mapping(w_j_original, j)

                value[:, :] += x_erp_up_j * mask_x_j
                count[:, :] += mask_x_j
            
            # save count map
            count_ = count / count.max()
            count_ = ToPILImage()(count_.cpu()[0][0])
            count_.save(f'/{save_dir}/{i+1:0>2}/count.png')

            # 7) Aggregate on canonical space
            x_erp_up = value / (count + 1e-8)
            
            ### 0.1.0 - SyncTweedies

            # 8) forward mapping x -> w^0
            w0_js = self.forward_mapping(x_erp_up)
            
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