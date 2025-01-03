
from base import DiffusionPipeBase
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

class CustomStableDiffusionPipeline(DiffusionPipeBase):
    __slots__ = ["device", "pipe", "resolution_factor", "vae", "tokenizer", "text_encoder", "unet", "scheduler"]
    def __init__(self, hf_key, device, half_precision=False):
        super().__init__()
        ddim = DDIMScheduler.from_pretrained(hf_key, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(hf_key, scheduler=ddim, torch_dtype=(torch.float16 if half_precision else torch.float32))
        pipe = pipe.to(device)
        self.vae, self.text_encoder, self.tokenizer, self.unet, self.scheduler, _, _, _ =  pipe.components.values()
        self.resolution_factor = 8
        self.device = device
        print(f'[INFO] loaded diffusion pipes!')
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        text_embeds = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeds

    @torch.no_grad()
    def encode_images(self, imgs):
        imgs = (imgs - 0.5) * 2
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        # Decoding
        latents = 1 / 0.18215 * latents
        latents = self.vae.decode(latents).sample
        # Post-processing
        latents = (latents / 2 + 0.5).clamp(0, 1)    
        return latents

    @torch.no_grad()
    def noise_guidance(self, noise_pred, guidance_scale):
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred
