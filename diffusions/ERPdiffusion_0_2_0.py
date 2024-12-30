import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torchvision.transforms import ToPILImage

from diffusions import ERPDiffusion_0_1_1
from geometry import make_coord, gridy2x_erp2pers


class ERPDiffusion_0_2_0(ERPDiffusion_0_1_1):
    """ MultiDiffusion w/ DeepFloyd-IF
    """
    
    def __init__(self, 
                 device, 
                 hf_key="DeepFloyd/IF-I-M-v1.0", 
                 fov=90,
                 views=[(0, 0), (45, 0)],
                 half_precision=False
                 ):
        if hf_key != "DeepFloyd/IF-I-M-v1.0":
            hf_key = "DeepFloyd/IF-I-M-v1.0"
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
                 height=128, width=256, 
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
        x_erp = torch.randn((1, self.unet.in_channels, height // self.resolution_factor, width // self.resolution_factor), device=self.device)

        # 2) Conditional Upsampling x^T_ERP
        x_erp_up = self.cond_noise_sampling(x_erp)

        # 3) Discrete warping x^T_ERP to {w^T_i}_i=1:N
        w_js = self.discrete_warping(x_erp_up, hw=(64, 64))
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
                noise_pred_uncond, _ = noise_pred_uncond.split(unet_input.shape[1], dim=1)
                noise_pred_cond, predicted_variance = noise_pred_cond.split(unet_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
                noise_pred, _ = noise_pred.split(unet_input.shape[1], dim=1)

                # 5) Get w'^t-1_j
                w_j_denoised = self.scheduler.step(noise_pred, t, w_j)['prev_sample']

                # 6) inverse mapping w -> x
                x_erp_up_j, mask_x_j = self.img_j_to_erp(w_j_denoised, j, erp_HW=(height, width))

                value[:, :] += x_erp_up_j * mask_x_j
                count[:, :] += mask_x_j
            
            # save count map
            count_ = count / count.max()
            count_ = ToPILImage()(count_.cpu()[0][0])
            count_.save(f'/{save_dir}/{i+1:0>2}/count.png')

            # 7) Aggregate on canonical space
            x_erp_up = value / (count + 1e-8)
            ToPILImage()(x_erp_up[0].cpu()).save(f"/{save_dir}/{i+1:0>2}/erp.png")

            # 8) forward mapping x -> w
            w_js = self.erp_to_img_j(x_erp_up, img_j_hw=(64, 64))
            buffer_imgs = self.decode_and_save(i, w_js, save_dir, buffer_imgs)
        
        return buffer_imgs