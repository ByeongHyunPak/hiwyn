import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from torchvision.transforms import ToPILImage

# MODEL_TYPE_STABLE_DIFFUSION, MODEL_TYPE_DEEPFLOYD
from geometry import generate_views, prepare_erp_pers_matching, erp_to_img_j, img_j_to_erp

from torch.cuda.amp import autocast

import gc

from base import CustomDeepfloydIFPipeline, CustomStableDiffusionPipeline



""" 
!python main.py --hf_key "DeepFloyd/IF-I-M-v1.0" --model "DualERPDiffusion_0_0_0" --hw 64 128  --theta_range 0 360 --num_theta 3 6 6 3 --phi_range -45 45 --num_phi 4 --fov 90
!python main.py --hf_key "DeepFloyd/IF-I-M-v1.0" --model "DualERPDiffusion_0_0_0" --hw 64 128  --theta_range 0 360 --num_theta 1 5 5 1 --phi_range -90 90 --num_phi 4 --fov 90
"""

class ERPDiffusionDualBranch(nn.Module):

    def __init__(self,
                 pipe,
                 device,
                 fov=90,
                 views=[(0, 0)],
                 half_precision=True
                 ): 
        super().__init__()
        self.device = device
        self.fov = fov
        self.views = views
        self.up_level = 3
    
        self.pipe = pipe
        self.half_precision = half_precision
        # ddim = DDIMScheduler.from_pretrained("DeepFloyd/IF-II-M-v1.0", subfolder="scheduler")
        # pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-M-v1.0", scheduler=ddim, variant="fp16", torch_dtype=(torch.float16 if half_precision else torch.float32))
        # pipe = pipe.to("cuda")

    @torch.no_grad()
    def stage_2(self,
                w0s,
                prompt_embeds,
                height=512, width=1024,
                num_inference_steps=100,
                guidance_scale=7.0,
                noise_level=50):
        self.pipe.unet.cpu()
        del self.pipe.unet
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        pipe = CustomDeepfloydIFPipeline("DeepFloyd/IF-II-M-v1.0", self.device, self.half_precision)
        self.pipe = pipe.to("cuda")
        
        del self.pipe.tokenizer
        self.pipe.text_encoder.cpu()
        del self.pipe.text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # https://github.com/dangeng/visual_anagrams/blob/main/visual_anagrams/samplers.py#L145
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_superresolution.py#L606
        
        h = w = self.pipe.unet.config.sample_size # 256

        # Get noisy images
        _, wts = self.pipe.sample_initial_noises(self.views, self.fov, height, width, h, w)

        # Define Pers. fusion map
        value = torch.zeros((1, 3, height, width), device=w0s[0].device)
        count = torch.zeros((1, 1, height, width), device=w0s[0].device)

        # Prepare ERP-Pers. projection
        self.erp2pers_pairs, self.pers2erp_grids = prepare_erp_pers_matching(self.views, self.fov, self.device, erp_img_hw=(height, width), pers_img_hw=(h, w))

        # Prepare upscaled image and noise level
        # w0s = [self.pipe.preprocess_image(w0, num_images_per_promt, self.device) for w0 in w0s]
        w0s = [self.pipe.encode_images(w0) for w0 in w0s]
        w0s = [F.interpolate(w0, (h, w), mode="bilinear", align_corners=True) for w0 in w0s]

        noise_level = torch.tensor([noise_level] * w0s[0].shape[0], device=w0s[0].device)
        w0s = [self.pipe.image_noising_scheduler.add_noise(w0, noise, timesteps=noise_level)
               for w0, noise in zip(w0s, wts)] # TODO: resample noise with other generator, not using wts
    
        # Condition on noise level, for each model input
        noise_level = torch.cat([noise_level] * 2)

        # set scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(tqdm(self.pipe.scheduler.timesteps)):

            os.makedirs(f"{self.save_dir}/stage2/{i+1:0>3}/", exist_ok=True)

            value.zero_(); count.zero_()

            # denoising each pers. view
            wts_original = []
            for j in range(len(self.views)):
                wt, w0 = wts[j], w0s[j]
                model_input = torch.cat([wt, w0], dim=1)
                model_input = torch.cat([model_input] * 2)
                model_input = self.pipe.scheduler.scale_model_input(model_input, t)
                
                # predict the noise residual
                with autocast():
                    noise_pred = self.pipe.unet(
                        model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        class_labels=noise_level,
                        cross_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                # class-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # ddim step
                wt_ddim_output = self.pipe.scheduler.step(noise_pred, t, wt)
                wts_original.append(wt_ddim_output['pred_original_sample'])
            
            # fuse every wts_original
            wts_original_img = [self.pipe.decode_latents(wj) for wj in wts_original]
            for j, w0_img in enumerate(wts_original_img):
                theta, phi = self.views[j]
                ToPILImage()(w0_img[0].cpu()).save(f"{self.save_dir}/stage2/{i+1:0>3}/w0_{theta}_{phi}.png")
            
            # Aggregate each Pers. original image on ERP grid
            wts_erp_img = self.aggregate_pers_imgs_on_erp(wts_original_img, value, count)
            ToPILImage()(wts_erp_img[0].cpu()).save(f"{self.save_dir}/stage2/{i+1:0>3}/w0.png")

            # Update wt w/ fused erp_img
            wts_original_img = erp_to_img_j(wts_erp_img, self.pers2erp_grids)
            wts_original = [self.pipe.encode_images(wj) for wj in wts_original_img]
            wts = [self.pipe.get_updated_noise(wt, w0, t) for wt, w0 in zip(wts, wts_original)]

            wts_img = []
            for j, wt in enumerate(wts):
                theta, phi = self.views[j]
                wt_img = pipe.decode_latents(wt)
                ToPILImage()(wt_img[0].cpu()).save(f"{self.save_dir}/stage2/{i+1:0>3}/wt_{theta}_{phi}.png")
                wts_img.append(wt_img)
        
        # save final erp image
        wts_erp_img = self.aggregate_pers_imgs_on_erp(wts_img, value, count)
        ToPILImage()(wts_erp_img[0].cpu()).save(f"{self.save_dir}/stage2/{i+1:0>3}/final_erp.png")

        return wts_img

    @torch.no_grad()
    def text2erp(self,
                 prompts,
                 negative_prompts='',
                 height=64, width=128,
                 num_inference_steps=50,
                 guidance_scale=7.5, # 7.0 in visual_anagrams for DeepFloyd-IF
                 save_dir="imgs/",
                 circular_padding=True,
                 spot_diffusion=False,
                 global_context_weight=0.5):

        self.save_dir = save_dir
        
        # setting dimensions
        H = height // self.pipe.resolution_factor # ERP noise h
        W = width // self.pipe.resolution_factor # ERP noise w
        h, w = 64, 64 # Pers. noise hw (= unet input hw)

        # Check if resolution is valid for Stable Diffusion
        if isinstance(self.pipe, CustomStableDiffusionPipeline):
            assert (height >= 512) & (width >= 512), \
                ValueError(f"Stable Diffusion's output must have over 512 pixels. Now ({width, height}).")

        # Get text_embeds and initial noises
        pers_text_embeds, erp_text_embeds = self.prepare_text_embeds(prompts, negative_prompts)
        # self.pipe.text_encoder.cpu()
        del self.pipe.tokenizer
        del self.pipe.text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        zt, wts = self.pipe.sample_initial_noises(self.views, self.fov, H, W, h, w)

        # Get ERP branch's noise crop positions
        zt_width = W//4 + W + W//4 if circular_padding else W # ERP noise width
        zt_stride = (8, 64) if spot_diffusion else (8, 8) # stride_hw for ERP noise
        zt_views = generate_views(H, zt_width, stride=zt_stride)

        # Define ERP fusion map
        value_z = torch.zeros((1, self.pipe.unet.config.in_channels, H, zt_width), device=zt.device)
        count_z = torch.zeros((1, 1, H, zt_width), device=zt.device)

        # Define Pers. fusion map
        value_w = torch.zeros((1, 3, height, width), device=zt.device)
        count_w = torch.zeros((1, 1, height, width), device=zt.device)

        # Prepare ERP-Pers. projection 
        erp_img_hw = (height, width)
        pers_img_hw = (64*self.pipe.resolution_factor, 64*self.pipe.resolution_factor)
        self.erp2pers_pairs, self.pers2erp_grids = prepare_erp_pers_matching(self.views, self.fov, self.device, erp_img_hw, pers_img_hw)

        # set scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(tqdm(self.pipe.scheduler.timesteps)):

            os.makedirs(f"{save_dir}/stage1/{i+1:0>2}/erp/", exist_ok=True)
            os.makedirs(f"{save_dir}/stage1/{i+1:0>2}/pers/", exist_ok=True)

            value_z.zero_(); count_z.zero_()
            value_w.zero_(); count_w.zero_()

            # Get each branch's ddim output
            zt_ddim_output = self.erp_denoising_branch(zt, erp_text_embeds, guidance_scale, t, zt_views, value_z, count_z, spot_diffusion, circular_padding)
            wts_ddim_outputs = self.pers_denoising_branch(wts, pers_text_embeds, guidance_scale, t)
            
            # Tweedie's formula: z0|t & w0|t
            zt_original = zt_ddim_output['pred_original_sample']
            wts_original = [wj['pred_original_sample'] for wj in wts_ddim_outputs]

            zt_original_img = self.pipe.decode_latents(zt_original)
            ToPILImage()(zt_original_img[0].cpu()).save(f"{save_dir}/stage1/{i+1:0>2}/erp/z0.png")

            wts_original_img = [self.pipe.decode_latents(wj) for wj in wts_original]
            for j, w0_img in enumerate(wts_original_img):
                theta, phi = self.views[j]
                ToPILImage()(w0_img[0].cpu()).save(f"{save_dir}/stage1/{i+1:0>2}/pers/w0_{theta}_{phi}.png")

            # Aggregate each Pers. original image on ERP grid
            wts_erp_img = self.aggregate_pers_imgs_on_erp(wts_original_img, value_w, count_w)
            ToPILImage()(wts_erp_img[0].cpu()).save(f"{save_dir}/stage1/{i+1:0>2}/pers/w0.png")

            # Fuse zt_original_img & wts_erp_img
            fused_erp_img = global_context_weight * zt_original_img + (1 - global_context_weight) * wts_erp_img
            ToPILImage()(fused_erp_img[0].cpu()).save(f"{save_dir}/stage1/{i+1:0>2}/fused_erp_img.png")

            # Update zt w/ fused erp_img
            zt_original = self.pipe.encode_images(fused_erp_img)
            zt = self.pipe.get_updated_noise(zt, zt_original, t)

            # Update wt w/ fused erp_img
            wts_original_img = erp_to_img_j(fused_erp_img, self.pers2erp_grids)
            wts_original = [self.pipe.encode_images(wj) for wj in wts_original_img]
            wts = [self.pipe.get_updated_noise(wt, w0, t) for wt, w0 in zip(wts, wts_original)]

            # save zt &  wt images
            zt_img = self.pipe.decode_latents(zt)
            ToPILImage()(zt_img[0].cpu()).save(f"{save_dir}/stage1/{i+1:0>2}/erp/zt.png")

            wts_img = []
            for j, wt in enumerate(wts):
                theta, phi = self.views[j]
                wt_img = self.pipe.decode_latents(wt)
                ToPILImage()(wt_img[0].cpu()).save(f"{save_dir}/stage1/{i+1:0>2}/pers/wt_{theta}_{phi}.png")
                wts_img.append(wt_img)
        
        # save final erp image
        wts_erp_img = self.aggregate_pers_imgs_on_erp(wts_img, value_w, count_w)
        ToPILImage()(wts_erp_img[0].cpu()).save(f"{self.save_dir}/stage1/{i+1:0>2}/final_erp.png")

        # if isinstance(self.pipe, CustomDeepfloydIFPipeline):
        #     wts_img = self.stage_2(
        #         w0s=wts_img,
        #         prompt_embeds=pers_text_embeds,
        #     )

        return wts_img

    @torch.no_grad()
    def aggregate_pers_imgs_on_erp(self,
                                   wts_original_img,
                                   value_w,
                                   count_w):
        
        B, C, H, W = value_w.shape

        for j, wt_img in enumerate(wts_original_img):
            per2erp_pixel_value, pers2erp_pixel_index = img_j_to_erp(wt_img, self.erp2pers_pairs[j])
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
        count_w.save(f"{self.save_dir}/stage1/pers_branch_cnt.png")

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
            wt_model_input = self.pipe.scheduler.scale_model_input(wt_model_input)

            ### unet noise prediction
            noise_pred = self.pipe.unet_autocast(wt_model_input, t, encoder_hidden_states=text_embeds)['sample']

            ### perform guidance
            noise_pred = self.pipe.noise_guidance(noise_pred, guidance_scale)

            wt_ddim_output = self.pipe.scheduler.step(noise_pred, t, wt)
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
            zt_model_input = self.pipe.scheduler.scale_model_input(zt_model_input)

            ### unet noise prediction
            noise_pred = self.pipe.unet_autocast(zt_model_input, t, encoder_hidden_states=text_embeds)

            ### perform guidance
            noise_pred = self.pipe.noise_guidance(noise_pred, guidance_scale)

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
        zt_ddim_output = self.pipe.scheduler.step(zt_noise_pred, t, zt)
        
        ### count map save
        count_z = count_z / count_z.max()
        count_z = ToPILImage()(count_z.cpu()[0][0])
        count_z.save(f"{self.save_dir}/stage1/erp_branch_cnt.png")

        return zt_ddim_output

    
    @torch.no_grad()
    def prepare_text_embeds(self, prompts, negative_prompts):
        assert isinstance(prompts, str), ValueError("prompts mush be str.")
        ### for pers. branch text_embeds
        prompts = [prompts]
        negative_prompts = [negative_prompts] if isinstance(negative_prompts, str) else negative_prompts
        pers_text_embeds = self.pipe.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]
        ### for erp branch text_embeds
        erp_prompts = f"360-degree panoramic image, {prompts}"
        erp_prompts = [erp_prompts]
        erp_text_embeds = self.pipe.get_text_embeds(erp_prompts, negative_prompts)  # [2, 77, 768]
        return pers_text_embeds, erp_text_embeds
