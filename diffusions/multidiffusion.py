from transformers import CLIPTextModel, CLIPTokenizer, logging, T5Tokenizer, T5EncoderModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from tqdm import tqdm

from diffusers import DiffusionPipeline, StableDiffusionPipeline

from base import CustomDeepfloydIFPipeline, CustomStableDiffusionPipeline

from geometry import generate_views

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

class MultiDiffusion(nn.Module):
    def __init__(self, pipe, device, **kwargs):
        super().__init__()

        self.device = device
        self.pipe = pipe

    @torch.no_grad()
    def text2panorama(self, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50,
                      guidance_scale=7.5, visualize_intermidiates=False, save_dir=None, circular_padding=True):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.pipe.get_text_embeds(prompts, negative_prompts)

        # Define panorama grid and get views
        latent = torch.randn((1, self.pipe.unet.in_channels, height // self.pipe.resolution_factor, width // self.pipe.resolution_factor), device=self.device)

        if circular_padding:
            w = width // self.pipe.resolution_factor
            latent = torch.cat((latent[:,:,:,-w//4:], latent, latent[:,:,:,:w//4]), dim=-1) # - circular padding
            views = generate_views(height, width + width//2)
        else:
            views = generate_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.pipe.scheduler.set_timesteps(num_inference_steps)

        if hasattr(self.pipe.scheduler, "set_begin_index"):
            self.pipe.scheduler.set_begin_index(0)

        with torch.autocast('cuda'):

            if visualize_intermidiates is True:
                intermidiate_imgs = []
                
            for i, t in enumerate(tqdm(self.pipe.scheduler.timesteps)):
                count.zero_()
                value.zero_()

                for h_start, h_end, w_start, w_end in views:
                    # TODO we can support batches, and pass multiple views at once to the unet
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end]

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latent_view] * 2)
                    latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t) if isinstance(self.pipe, CustomDeepfloydIFPipeline) else latent_model_input
                    # predict the noise residual
                    noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                    
                    # perform guidance
                    if isinstance(self.pipe, CustomStableDiffusionPipeline): # TODO(jw): migrate this logic into pipes
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    elif isinstance(self.pipe, CustomDeepfloydIFPipeline):
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
                        noise_pred_cond, predicted_variance = noise_pred_cond.split(latent_model_input.shape[1], dim=1)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
                        noise_pred, _ = noise_pred.split(latent_model_input.shape[1], dim=1)

                    # compute the denoising step with the reference model
                    latents_view_denoised = self.pipe.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                    value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                # take the MultiDiffusion step
                latent = torch.where(count > 0, value / count, value)

                if circular_padding:
                    latent = latent[:,:,:,w//4:-w//4]

                # visualize intermidiate timesteps
                if visualize_intermidiates is True:
                    imgs = self.pipe.decode_latents(latent)  # [1, 3, 512, 512]
                    img = T.ToPILImage()(imgs[0].cpu())
                    intermidiate_imgs.append((i, img))
                
                if circular_padding:
                    latent = torch.cat((latent[:,:,:,-w//4:], latent, latent[:,:,:,:w//4]), dim=-1) # - circular padding

        if circular_padding:
            latent = latent[:,:,:,w//4:-w//4]

        # Img latents -> imgs
        imgs = self.pipe.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        if save_dir is not None:
            img.save(save_dir + "/result.png")
        if visualize_intermidiates is True:
            intermidiate_imgs.append((len(intermidiate_imgs), img))
            return intermidiate_imgs
        else:
            return [img]