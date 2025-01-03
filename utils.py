from geometry import make_coord, gridy2x_erp2pers, gridy2x_pers2erp

import torch
import torch.nn.functional as F

MODEL_TYPE_STABLE_DIFFUSION, MODEL_TYPE_DEEPFLOYD = "stable-diffusion", "DeepFloyd"

def load_models(hf_key):
    if MODEL_TYPE_STABLE_DIFFUSION in hf_key:
        mode = MODEL_TYPE_STABLE_DIFFUSION
    elif MODEL_TYPE_DEEPFLOYD in hf_key:
        mode = MODEL_TYPE_DEEPFLOYD
    else:
        ValueError("Unknown model (available model: stabilityai/stable-diffusion-2-1-base|stabilityai/stable-diffusion-2-base|runwayml/stable-diffusion-v1-5|DeepFloyd/IF-I-M-v1.0)")

    ddim = DDIMScheduler.from_pretrained(hf_key, subfolder="scheduler")

    # print(pipe.components.keys()) # 'vae', 'text_encoder', 'tokenizer', 'unet', 'scheduler', 'safety_checker', 'feature_extractor', 'image_encoder
    if mode == MODEL_TYPE_STABLE_DIFFUSION:
        pipe = StableDiffusionPipeline.from_pretrained(hf_key, scheduler=ddim, torch_dtype=(torch.float16 if half_precision else torch.float32))
        pipe = pipe.to("cuda")
        self.vae, self.text_encoder, self.tokenizer, self.unet, self.scheduler, _, _, _ =  pipe.components.values()
        self.encode_prompt = lambda prompt, negative_prompt: self.stable_diffusion_encode_prompt(prompt, negative_prompt)
        self.resolution_factor = 8
    elif mode == MODEL_TYPE_DEEPFLOYD:
        pipe = DiffusionPipeline.from_pretrained(hf_key, scheduler=ddim, variant="fp16", torch_dtype=(torch.float16 if half_precision else torch.float32))
        pipe = pipe.to("cuda")
        self.tokenizer, self.text_encoder, self.unet, self.scheduler, _, _, _ =  pipe.components.values()
        self.encode_prompt = lambda prompt, negative_prompt: pipe.encode_prompt(prompt, do_classifier_free_guidance=True, num_images_per_prompt=1, device=self.device, negative_prompt=negative_prompt)
        self.resolution_factor = 1

    print(f'[INFO] loaded diffusion pipes!')
    
    
#########################################################################################
######################################### Noise #########################################
#########################################################################################
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

#########################################################################################
########################################## VAE ##########################################
#########################################################################################
@torch.no_grad()
def encode_images(vae, imgs):
    imgs = (imgs - 0.5) * 2
    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * 0.18215
    return latents




#########################################################################################
######################################## Warping ########################################
#########################################################################################
@torch.no_grad()
def img_j_to_erp(img_j, erp2pers_pair):
    erp2pers_grid, erp_indices = erp2pers_pair # (_, D, _, 2), (D,)

    img_j_to_erp_img = F.grid_sample(
        img_j, # (1, C, h, w)
        erp2pers_grid, # (_, D, _, 2)
        mode="bilinear", align_corners=False,
        padding_mode="reflection"
    ) # (_, C, D, 1)

    img_j_to_erp_img.clamp_(0, 1)

    img_j_to_erp_img = img_j_to_erp_img.view(img_j.shape[1], -1) # (_, C, D, 1) -> (C, D)

    return img_j_to_erp_img, erp_indices # (C, D), (D,); value, indices

@torch.no_grad()
def erp_to_img_j(erp_img, pers2erp_grids):
    pers2erp_grids_input = torch.stack(pers2erp_grids).squeeze(1) 

    img_js = F.grid_sample(
        erp_img.expand(len(self.views), -1, -1 ,-1),
        pers2erp_grids_input,
        padding_mode="reflection",
        mode="bilinear", align_corners=False)
    
    img_js.clamp_(0, 1)
    
    img_js = img_js.unsqueeze(1)
    
    return list(torch.unbind(img_js, dim=0))

'''
Prepare each view's Perspective coordinates w.r.t. ERP indices (erp2pers_grid) and corresponding ERP indices (erp_indices)
'''
@torch.no_grad()
def prepare_erp_pers_matching(erpHW, imgHW=(512, 512)):
    '''
    erp2pers_pairs: list of (erp2pers_grid, erp_indices)
    '''

    erp2pers_pairs = [] 
    pers2erp_grids = []
    
    erp_grid = make_coord(erpHW, flatten=False).to(self.device) # (H, W, 2); ERP coordinates
    pers_grid = make_coord(imgHW, flatten=False).to(self.device) # (h, w, 2)

    for theta, phi in self.views:  
        ### ERP2PERS ###      
        erp2pers_grid, valid_mask = gridy2x_erp2pers(gridy=erp_grid,
            HWy=erpHW, HWx=imgHW, THETA=theta, PHI=phi, FOVy=360, FOVx=90) # (H*W, 2), (H*W)
    
        # Filter valid indices
        erp_indices = torch.arange(0, erpHW[0]*erpHW[1]).to(self.device) # (H*W,)
        erp_indices = erp_indices[valid_mask.bool()].long() # sample (D,) from (H*W,)

        erp2pers_grid = erp2pers_grid[valid_mask.bool()] # sample (D, 2) from (H*W, 2)
        erp2pers_grid = erp2pers_grid.unsqueeze(1).unsqueeze(0) # (D, 2) -> (1, D, 1, 2)
        erp2pers_grid = erp2pers_grid.flip(-1) # x <-> y

        erp2pers_pairs.append((erp2pers_grid, erp_indices))

        ### PERS2ERP ###
        pers2erp_grid, valid_mask = gridy2x_pers2erp(gridy=pers_grid,
            HWy=imgHW, HWx=erpHW, THETA=theta, PHI=phi, FOVy=self.fov, FOVx=360)
        pers2erp_grid = pers2erp_grid.view(*imgHW, 2).unsqueeze(0).flip(-1)
        pers2erp_grids.append(pers2erp_grid)

    return erp2pers_pairs, pers2erp_grids
