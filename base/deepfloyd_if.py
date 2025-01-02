from base import DiffusionUtilMixin
from diffusers import DiffusionPipeline, DDIMScheduler
import torch

class CustomDeepfloydIFPipeline(DiffusionUtilMixin):
    __slots__ = ["pipe", "resolution_factor"]
    def __init__(self, hf_key, device, half_precision=False):
        ddim = DDIMScheduler.from_pretrained(hf_key, subfolder="scheduler")
        pipe = DiffusionPipeline.from_pretrained(hf_key, scheduler=ddim, variant="fp16", torch_dtype=(torch.float16 if half_precision else torch.float32), device_map="balanced", low_cpu_mem_usage=True)
        pipe = pipe.to(device)
        # self.tokenizer, self.text_encoder, self.unet, self.scheduler, _, _, _ =  pipe.components.values()
        self.resolution_factor = 1
        print(f'[INFO] loaded diffusion pipes!')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
        max_length = 77

        prompt = [p.lower().strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        attention_mask = text_inputs.attention_mask.to(self.device)

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        dtype = self.text_encoder.dtype

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = None
        if negative_prompt is None:
            uncond_tokens = [""]
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif len(negative_prompt) != 1:
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {1}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=False)
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        attention_mask = uncond_input.attention_mask.to(self.device)

        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=self.device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(1, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes

        text_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]) 
        return text_embeds
        # or pipe.encode_prompt
    
    @torch.no_grad()
    def decode_latents(self, latents):
        # No Decoding
        # Post-processing
        latents = (latents / 2 + 0.5).clamp(0, 1)    
        return latents