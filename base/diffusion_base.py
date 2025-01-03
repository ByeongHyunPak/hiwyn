import os 
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from geometry import erp2pers_discrete_warping

class DiffusionPipeBase():
    __slots__ = ["up_level", "scheduler"]
    def __init__(self, half_precision):
        self.up_level = 3
        self.scheduler = None
        self.half_precision = half_precision

    @torch.no_grad()
    def unet_autocast(self, input, t, encoder_hidden_states):
        if self.half_precision:
            input = input.half()
            t = t.half()
            encoder_hidden_states = encoder_hidden_states.half()

        return self.unet(input, t, encoder_hidden_states=encoder_hidden_states)['sample'].float()

    def decode_latents(self, latent):
        NotImplementedError("decode_latents() is not defined in this pipe")

    def decode_and_save(self, step, views, latents, save_dir, buffer, tag='pers'):
        pers_imgs = []
        for k, latent in enumerate(latents):
            pers_img = self.decode_latents(latent)
            pers_imgs.append((views[k], pers_img)) # [(theta, phi), img]
        buffer.append((step+1, pers_imgs)) # [i+1, [(theta, phi), img]]
        
        if save_dir is not None:
            # save image
            if os.path.exists(f"{save_dir}/{step+1:0>2}") is False:
                os.mkdir(f"{save_dir}/{step+1:0>2}/")
            for v, im in pers_imgs:
                theta, phi = v
                im = ToPILImage()(im[0].cpu())
                im.save(f'/{save_dir}/{step+1:0>2}/{tag}_{theta}_{phi}.png')
        
        return buffer
    
    @torch.no_grad()
    def cond_noise_sampling(self, src_noise):
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

        return up_noise # sqrt(N_k) W(A_k): sub-pixel noise scaled with sqrt(N_k) ~ N(0, 1)

    @torch.no_grad()
    def sample_initial_noises(self, views, fov, H, W, h, w):
        zt = torch.randn((1, self.unet.config.in_channels, H//self.resolution_factor, W//self.resolution_factor), device=self.device) # 1) Sample initial zT (ERP noise)
        zt_up = self.cond_noise_sampling(zt) # 2) Conditional Upsampling zT
        wts = erp2pers_discrete_warping(views, zt_up, fov, (h, w)) # 3) Get wTs from zT_up by discrete warping
        return zt, wts
    
    @torch.no_grad()
    def get_xt_x0_coeff(self, t):
        prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        std_dev_t = 0.0
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
        xt_coeff = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) / torch.sqrt(1 - alpha_prod_t)
        x0_coeff = torch.sqrt(alpha_prod_t_prev) - torch.sqrt(alpha_prod_t) / torch.sqrt(1 - alpha_prod_t) * torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2)
        return xt_coeff, x0_coeff

    @torch.no_grad()
    def get_updated_noise(self, xt, x0, t):
        xt_coeff, x0_coeff = self.get_xt_x0_coeff(t)
        xt_new =  xt_coeff * xt + x0_coeff * x0
        return xt_new