from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from tqdm import tqdm

from diffusers import DiffusionPipeline, StableDiffusionPipeline

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

MODEL_TYPE_STABLE_DIFFUSION, MODEL_TYPE_DEEPFLOYD = "stable-diffusion", "DeepFloyd"
class MultiDiffusion_2(nn.Module):

    """ SyncTweedies
    """

    def __init__(self, device, hf_key="stabilityai/stable-diffusion-2-base", half_precision=False, **kwargs):
        super().__init__()

        self.device = device

        if MODEL_TYPE_STABLE_DIFFUSION in hf_key:
            self.mode = MODEL_TYPE_STABLE_DIFFUSION
        elif MODEL_TYPE_DEEPFLOYD in hf_key:
            self.mode = MODEL_TYPE_DEEPFLOYD
        else:
            ValueError("Unknown model (available model: stabilityai/stable-diffusion-2-1-base|stabilityai/stable-diffusion-2-base|runwayml/stable-diffusion-v1-5|DeepFloyd/IF-I-M-v1.0)")

        ddim = DDIMScheduler.from_pretrained(hf_key, subfolder="scheduler")

        # print(pipe.components.keys()) # 'vae', 'text_encoder', 'tokenizer', 'unet', 'scheduler', 'safety_checker', 'feature_extractor', 'image_encoder
        if self.mode == MODEL_TYPE_STABLE_DIFFUSION:
            pipe = StableDiffusionPipeline.from_pretrained(hf_key, scheduler=ddim, torch_dtype=(torch.float16 if half_precision else torch.float32))
            pipe = pipe.to("cuda")
            self.vae, self.text_encoder, self.tokenizer, self.unet, self.scheduler, _, _, _ =  pipe.components.values()
            self.encode_prompt = lambda prompt, negative_prompt: self.stable_diffusion_encode_prompt(prompt, negative_prompt)
            self.resolution_factor = 8
        elif self.mode == MODEL_TYPE_DEEPFLOYD:
            pipe = DiffusionPipeline.from_pretrained(hf_key, scheduler=ddim, variant="fp16", torch_dtype=(torch.float16 if half_precision else torch.float32))
            pipe = pipe.to("cuda")
            self.tokenizer, self.text_encoder, self.unet, self.scheduler, _, _, _ =  pipe.components.values()
            self.encode_prompt = lambda prompt, negative_prompt: pipe.encode_prompt(prompt, do_classifier_free_guidance=True, num_images_per_prompt=1, device=self.device, negative_prompt=negative_prompt)
            self.resolution_factor = 1

        print(f'[INFO] loaded diffusion pipes!')
    
    @torch.no_grad()
    def stable_diffusion_encode_prompt(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        return uncond_embeddings, text_embeddings
    
    @torch.no_grad()
    def get_views(self, panorama_height, panorama_width, window_size=64, stride=8):
        panorama_height /= self.resolution_factor
        panorama_width /= self.resolution_factor
        num_blocks_height = (panorama_height - window_size) // stride + 1
        num_blocks_width = (panorama_width - window_size) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
        return views

    @torch.no_grad()
    def latent2image(self, latents):
        # Decoding
        if self.mode == MODEL_TYPE_STABLE_DIFFUSION: 
            latents = 1 / 0.18215 * latents
            latents = self.vae.decode(latents).sample
        # Post-processing
        latents = (latents / 2 + 0.5).clamp(0, 1)    
        # if self.mode == MODEL_TYPE_DEEPFLOYD:
        #     latents = latents.permute(0, 2, 3, 1).float()
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        return self.latent2image(latents)

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompts, negative_prompts)  # [2, 77, 768]
        if self.mode == MODEL_TYPE_DEEPFLOYD:
            text_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]) 
        elif self.mode == MODEL_TYPE_STABLE_DIFFUSION: 
            text_embeds = torch.cat([prompt_embeds, negative_prompt_embeds])
        return text_embeds
    
    @torch.no_grad()
    def text2panorama(self, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50,
                      guidance_scale=7.5, visualize_intermidiates=False, save_dir=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)

        # Define panorama grid and get views
        latent = torch.randn((1, self.unet.in_channels, height // self.resolution_factor, width // self.resolution_factor), device=self.device)
        views = self.get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        with torch.autocast('cuda'):

            latents = []
            for h_start, h_end, w_start, w_end in views:
                latent_view = latent[:, :, h_start:h_end, w_start:w_end]
                latents.append(latent_view)

            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                count.zero_()
                value.zero_()

                cnt = 0
                for latent_view in latents:
                    
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latent_view] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) if self.mode == MODEL_TYPE_DEEPFLOYD else latent_model_input
                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                    
                    # perform guidance
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
                    noise_pred_cond, predicted_variance = noise_pred_cond.split(latent_model_input.shape[1], dim=1)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
                    noise_pred, _ = noise_pred.split(latent_model_input.shape[1], dim=1)

                    # compute the denoising step with the reference model
                    ### syncTweedies
                    ddim_output = self.scheduler.step(noise_pred, t, latent_view)
                    x_j_denoised = ddim_output['prev_sample']
                    x_j_original = ddim_output['pred_original_sample']

                    x_j_original_ = self.latent2image(x_j_original)
                    T.ToPILImage()(x_j_original_[0].cpu()).save(f"{save_dir}/{i:0>2}_{cnt}_x_original.png")
                    cnt += 1

                    value[:, :, h_start:h_end, w_start:w_end] += x_j_original
                    count[:, :, h_start:h_end, w_start:w_end] += 1                   

                x_original = torch.where(count > 0, value / count, value)
                x_originals = []
                for h_start, h_end, w_start, w_end in views:
                    x_originals.append(x_original[:, :, h_start:h_end, w_start:w_end])

                latents_new = []

                cnt = 0
                for latent_view, x_orig in zip(latents, x_originals):
                    timestep = t
                    prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

                    alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
                    alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

                    beta_prod_t = 1 - alpha_prod_t

                    variance = self.scheduler._get_variance(timestep, prev_timestep)
                    std_dev_t = 0.0

                    xt_coeff = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) / torch.sqrt(1 - alpha_prod_t)
                    x0_coeff = torch.sqrt(alpha_prod_t_prev) - torch.sqrt(alpha_prod_t) / torch.sqrt(1 - alpha_prod_t) * torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2)
                    
                    w_new_j = xt_coeff * latent_view + x0_coeff * x_orig
                    w_new_j_ = self.latent2image(w_new_j)
                    T.ToPILImage()(w_new_j_[0].cpu()).save(f"{save_dir}/{i:0>2}_{cnt}_w_new_j.png")
                    cnt += 1

                    latents_new.append(w_new_j)
                
                latents = latents_new

        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)
        
        for j, view in enumerate(views):
            h_start, h_end, w_start, w_end = view
            value[:, :, h_start:h_end, w_start:w_end] += latents[j]
            count[:, :, h_start:h_end, w_start:w_end] += 1   
        latent = torch.where(count > 0, value / count, value)

        # Img latents -> imgs
        imgs = self.latent2image(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        if save_dir is not None:
            img.save(save_dir + "/result.png")
            return [img]