import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torchvision.transforms import ToPILImage

from diffusions import MultiDiffusion
from geometry import make_coord, gridy2x_erp2pers


class ERPDiffusion_0_0_0(MultiDiffusion):
    """ MultiDiffusion + HIWYN (conditional upsampling & discrete warping)
    """
    
    def __init__(self, 
                 device, 
                 hf_key=None, 
                 fov=90,
                 views=[(0, 0), (45, 0)],
                 half_precision=False
                 ):
        super().__init__(device, hf_key, half_precision)
        
        self.views = views
        self.fov = fov

        self.fin_v_nums = []
        self.indices = None
    
    def decode_and_save(self, i, latents, save_dir, buffer, tag='pers'):

        pers_imgs = []
        for k, latent in enumerate(latents):
            pers_img = self.decode_latents(latent)
            pers_imgs.append((self.views[k], pers_img)) # [(theta, phi), img]
        buffer.append((i+1, pers_imgs)) # [i+1, [(theta, phi), img]]
        
        if save_dir is not None:
            # save image
            if os.path.exists(f"{save_dir}/{i+1:0>2}") is False:
                os.mkdir(f"{save_dir}/{i+1:0>2}/")
            for v, im in pers_imgs:
                theta, phi = v
                im = ToPILImage()(im[0].cpu())
                im.save(f'/{save_dir}/{i+1:0>2}/{tag}_{theta}_{phi}.png')
        
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
    def discrete_warping(self, src_white_noise, hw=(512//8, 512//8), normalize_fin_v_val=True):
        B, C, H, W = src_white_noise.shape
        h, w = hw

        erp_grid = make_coord((H, W), flatten=False).to(self.device) # (H, W, 2)
        erp_up_noise_flat = src_white_noise.reshape(B*C, -1)

        pers_idx = torch.arange(1, h*w+1, device=self.device, dtype=torch.float32).view(1, h, w) # (1, h, w)
        ones_flat = torch.ones_like(erp_up_noise_flat[:1])

        tgts = []
        indices = []

        for theta, phi in self.views:
            erp2pers_grid, valid_mask = gridy2x_erp2pers(gridy=erp_grid,
                HWy=(H, W), HWx=hw, THETA=theta, PHI=phi, FOVy=360, FOVx=self.fov)
            erp2pers_grid = erp2pers_grid.view(H, W, 2)
            valid_mask = valid_mask.view(1, H, W)
            
            # Find nearest grid index of erp pixel on Pers. grid
            erp2pers_idx = F.grid_sample(
                pers_idx.unsqueeze(0),
                erp2pers_grid.unsqueeze(0).flip(-1),
                mode="nearest", align_corners=False)[0] # (1, H, W)
            erp2pers_idx *= valid_mask # non-mapped pixel has 0 value.
            erp2pers_idx = erp2pers_idx.to(torch.int64)

            ind_flat = erp2pers_idx.view(1, -1)

            # 5) Get warped target noise
            fin_v_val = torch.zeros(B*C, h*w+1, device=self.device) \
                .scatter_add_(1, index=ind_flat.repeat(B*C, 1), src=erp_up_noise_flat)[..., 1:]
            fin_v_num = torch.zeros(1, h*w+1, device=self.device) \
                .scatter_add_(1, index=ind_flat, src=ones_flat)[..., 1:]
            assert fin_v_num.min() != 0, ValueError(f"{theta},{phi}")

            if normalize_fin_v_val:
                final_values = fin_v_val / torch.sqrt(fin_v_num)
            else:
                final_values = fin_v_val
            tgt_warped_noise = final_values.reshape(B, C, h, w).float()
            tgt_warped_noise = tgt_warped_noise.to(self.device)
            
            tgts.append(tgt_warped_noise)
            indices.append(erp2pers_idx.reshape((H, W)))

            if len(self.fin_v_nums) < len(self.views):
                self.fin_v_nums.append(fin_v_num.reshape(h, w))
        
        self.indices = indices
        return tgts

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

                # 5) Get w'^t-1_j
                w_j_denoised = self.scheduler.step(noise_pred, t, w_j)['prev_sample']

                # 6) inverse mapping w -> x
                x_erp_up_j, mask_x_j = self.inverse_mapping(w_j_denoised, j)

                value[:, :] += x_erp_up_j * mask_x_j
                count[:, :] += mask_x_j
            
            # save count map
            count_ = count / count.max()
            count_ = ToPILImage()(count_.cpu()[0][0])
            count_.save(f'/{save_dir}/{i+1:0>2}/count.png')

            # 7) Aggregate on canonical space
            x_erp_up = value / (count + 1e-8)

            # 8) forward mapping x -> w
            w_js = self.forward_mapping(x_erp_up)
            buffer_imgs = self.decode_and_save(i, w_js, save_dir, buffer_imgs)
        
        return buffer_imgs

    def forward_mapping(self, x):
        tgts = self.discrete_warping(x, normalize_fin_v_val=False)
        return tgts

    def inverse_mapping(self, w_j, j):
        
        B, C, h, w = w_j.shape
        H, W  = self.indices[j].shape

        sub_x_j = torch.zeros(B*C, H*W, device=self.device)
        w_j_flat = w_j.reshape(B*C, -1) / self.fin_v_nums[j].view(1, -1)
        w_j_flat_pad = torch.zeros(B*C, h*w + 1, device=self.device)
        w_j_flat_pad[:, 1:] = w_j_flat

        ind_x2w_j = self.indices[j]
        mask_x_j = ind_x2w_j > 0
        
        sub_x_j[:, ...] = w_j_flat_pad[:, ind_x2w_j.flatten()]
        sub_x_j = sub_x_j.reshape(B, C, H, W)
        
        return sub_x_j, mask_x_j
        