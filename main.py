import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from torchvision.transforms import ToPILImage

import gc
import traceback
from diffusions import *

if __name__ == '__main__':

    seed_everything(2024)
    
    parser = argparse.ArgumentParser()

    # defaults
    parser.add_argument('--img_dir', type=str, default="imgs")
    parser.add_argument('--hf_key', type=str, default='stabilityai/stable-diffusion-2-base')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--hw', type=int, nargs=2, default=[1024, 2048])
    parser.add_argument('--half_precision', type=bool, default=True)
    
    # options
    parser.add_argument('--model', type=str, default='ERPDiffusion_0_1_1')
    parser.add_argument('--prompt', type=str, default="A photo of realistic cityscape of Florence")
    parser.add_argument('--theta_range', nargs=2, type=float, default=[0, 360])
    parser.add_argument('--num_theta', nargs="+", type=int, default=[3, 4, 6, 4, 3])
    parser.add_argument('--phi_range', nargs=2, type=float, default=[-45, 45])
    parser.add_argument('--num_phi', type=int, default=5)
    parser.add_argument('--fov', type=float, default=90)
    
    args = parser.parse_args()
    
    # set save_dir
    dir_name = args.img_dir
    base_dir = f'/content/{dir_name}'
    save_dir = f'{base_dir}/{args.prompt.split(" ")[0]}-{args.model}/'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # set view directions
    assert args.num_phi == len(args.num_theta)

    directions = []
    phis = np.linspace(*args.phi_range, args.num_phi, endpoint=True)
    print(">> View directions:")
    for i in range(args.num_phi):
        thetas = np.linspace(*args.theta_range, args.num_theta[i], endpoint=False)
        for theta in thetas:
            directions.append((theta, phis[i]))
        print(*directions[-args.num_theta[i]:])   

with torch.autocast(device_type='cuda', dtype=(torch.float16 if args.half_precision else torch.float32)):          
    try:
        H, W = args.hw
        model = globals()[f"{args.model}"]
        
        if "MultiDiffusion" in args.model:
            sd = model(device=torch.device('cuda'), hf_key=args.hf_key, half_precision=args.half_precision, fov=args.fov, views=directions) 
            outputs = sd.text2panorama(args.prompt, args.negative, height=H, width=W, num_inference_steps=args.steps, save_dir=save_dir)
        else:
            sd = model(device=torch.device('cuda'), hf_key=args.hf_key, half_precision=args.half_precision, fov=args.fov, views=directions) 
            outputs = sd.text2erp(
                args.prompt, args.negative, height=H, width=W, num_inference_steps=args.steps, save_dir=save_dir)
            
        del outputs
        del sd
        del model
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception:
        print(traceback.format_exc()) 
        del sd
        del model
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
        