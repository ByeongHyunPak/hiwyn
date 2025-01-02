from base import DiffusionPipeBase
from diffusers import DiffusionPipeline, DDIMScheduler
import torch

class CustomDeepfloydIFPipeline(DiffusionPipeBase):
    __slots__ = ["device", "encode_prompt", "pipe", "resolution_factor", "tokenizer", "text_encoder", "unet", "scheduler"]
    def __init__(self, hf_key, device, half_precision=False):
        super().__init__()
        ddim = DDIMScheduler.from_pretrained(hf_key, subfolder="scheduler")
        pipe = DiffusionPipeline.from_pretrained(hf_key, scheduler=ddim, variant="fp16", torch_dtype=(torch.float16 if half_precision else torch.float32))
        pipe = pipe.to(device)
        self.tokenizer, self.text_encoder, self.unet, self.scheduler, _, _, _ =  pipe.components.values()
        self.resolution_factor = 1
        self.device = device
        self.encode_prompt = lambda prompt, negative_prompt: pipe.encode_prompt(prompt, do_classifier_free_guidance=True, num_images_per_prompt=1, device=self.device, negative_prompt=negative_prompt)
        print(f'[INFO] loaded diffusion pipes!')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        text_embeddings, uncond_embeddings = self.encode_prompt(prompt, negative_prompt)
        return torch.cat([uncond_embeddings, text_embeddings])
        # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
        # max_length = 77

        # prompt = [p.lower().strip() for p in prompt]
        # text_inputs = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=max_length,
        #     truncation=True,
        #     add_special_tokens=True,
        #     return_tensors="pt",
        # )
        # text_input_ids = text_inputs.input_ids
        # untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        # attention_mask = text_inputs.attention_mask.to(self.device)

        # prompt_embeds = self.text_encoder(
        #     text_input_ids.to(self.device),
        #     attention_mask=attention_mask,
        # )
        # prompt_embeds = prompt_embeds[0]

        # dtype = self.text_encoder.dtype

        # prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)

        # bs_embed, seq_len, _ = prompt_embeds.shape

        # # duplicate text embeddings for each generation per prompt, using mps friendly method
        # prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        # prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        # # get unconditional embeddings for classifier free guidance
        # uncond_tokens = None
        # if negative_prompt is None:
        #     uncond_tokens = [""]
        # elif isinstance(negative_prompt, str):
        #     uncond_tokens = [negative_prompt]
        # elif len(negative_prompt) != 1:
        #     raise ValueError(
        #         f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
        #         f" {prompt} has batch size {1}. Please make sure that passed `negative_prompt` matches"
        #         " the batch size of `prompt`."
        #     )
        # else:
        #     uncond_tokens = negative_prompt

        # uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=False)
        # max_length = prompt_embeds.shape[1]
        # uncond_input = self.tokenizer(
        #     uncond_tokens,
        #     padding="max_length",
        #     max_length=max_length,
        #     truncation=True,
        #     return_attention_mask=True,
        #     add_special_tokens=True,
        #     return_tensors="pt",
        # )
        # attention_mask = uncond_input.attention_mask.to(self.device)

        # negative_prompt_embeds = self.text_encoder(
        #     uncond_input.input_ids.to(self.device),
        #     attention_mask=attention_mask,
        # )
        # negative_prompt_embeds = negative_prompt_embeds[0]

        # # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        # seq_len = negative_prompt_embeds.shape[1]

        # negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=self.device)

        # negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        # negative_prompt_embeds = negative_prompt_embeds.view(1, seq_len, -1)

        # # For classifier free guidance, we need to do two forward passes.
        # # Here we concatenate the unconditional and text embeddings into a single batch
        # # to avoid doing two forward passes

        # text_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]) 
        # return text_embeds
    
    @torch.no_grad()
    def encode_images(self, imgs):
        # No VAE in DeepFloyd/IF
        imgs = (imgs - 0.5) * 2
        return imgs

    @torch.no_grad()
    def decode_latents(self, latents):
        # No Decoding
        # Post-processing
        latents = (latents / 2 + 0.5).clamp(0, 1)    
        return latents
    
    @torch.no_grad()
    def noise_guidance(self, noise_pred, guidance_scale):
        channel = self.unet.config.in_channels
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred_uncond, _ = noise_pred_uncond.split(channel, dim=1)
        noise_pred_cond, predicted_variance = noise_pred_cond.split(channel, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
        noise_pred, _ = noise_pred.split(channel, dim=1)

        return noise_pred
    