import torch
import torch.nn.functional as F
import numpy as np

def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def rodrigues_torch(rvec):
    theta = torch.norm(rvec)
    if theta < torch.finfo(torch.float32).eps:
        rotation_mat = torch.eye(3, device=rvec.device, dtype=rvec.dtype)
    else:
        r = rvec / theta 
        I = torch.eye(3, device=rvec.device)
        
        r_rT = torch.outer(r, r)
        r_cross = torch.tensor([[0, -r[2], r[1]],
                                [r[2], 0, -r[0]],
                                [-r[1], r[0], 0]], device=rvec.device)
        rotation_mat = torch.cos(theta) * I + (1 - torch.cos(theta)) * r_rT + torch.sin(theta) * r_cross
    
    return rotation_mat


def gridy2x_pers2erp(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx):
    device = gridy.device
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx
    
    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    gridy[:, 0] *= np.tan(np.radians(hFOVy / 2.0))
    gridy[:, 1] *= np.tan(np.radians(wFOVy / 2.0))
    gridy = gridy.double().flip(-1)
    
    x0 = torch.ones(gridy.shape[0], 1, device=device)
    gridy = torch.cat((x0, gridy), dim=-1)
    gridy /= torch.norm(gridy, p=2, dim=-1, keepdim=True)
    
    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    gridy = torch.mm(R1, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R2, gridy.permute(1, 0)).permute(1, 0)

    ### sphere to gridx
    lat = torch.arcsin(gridy[:, 2]) / np.pi * 2
    lon = torch.atan2(gridy[:, 1] , gridy[:, 0]) / np.pi
    gridx = torch.stack((lat, lon), dim=-1)

    # masky
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]

    return gridx.to(torch.float32), mask.to(torch.float32)

def gridy2x_erp2pers(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx):
    device = gridy.device
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx

    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    lat = gridy[:, 0] * np.pi / 2
    lon = gridy[:, 1] * np.pi

    z0 = torch.sin(lat)
    y0 = torch.cos(lat) * torch.sin(lon)
    x0 = torch.cos(lat) * torch.cos(lon)
    gridy = torch.stack((x0, y0, z0), dim=-1).double()

    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    R1_inv = torch.inverse(R1)
    R2_inv = torch.inverse(R2)

    gridy = torch.mm(R2_inv, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R1_inv, gridy.permute(1, 0)).permute(1, 0)

    ### sphere to gridx
    z0 = (gridy[:, 2] / gridy[:, 0]) / np.tan(np.radians(FOVx / 2.0))
    y0 = (gridy[:, 1] / gridy[:, 0]) / np.tan(np.radians(FOVx / 2.0))
    gridx = torch.stack((z0, y0), dim=-1).float()

    # masky
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask *= torch.where(gridy[:, 0] < 0, 0, 1)

    return gridx.to(torch.float32), mask.to(torch.float32)

'''
Prepare each view's Perspective coordinates w.r.t. ERP indices (erp2pers_grid) and corresponding ERP indices (erp_indices)
'''
@torch.no_grad()
def prepare_erp_pers_matching(views, fov, device, erpHW, imgHW=(512, 512)):
    '''
    erp2pers_pairs: list of (erp2pers_grid, erp_indices)
    '''

    erp2pers_pairs = [] 
    pers2erp_grids = []
    
    erp_grid = make_coord(erpHW, flatten=False).to(device) # (H, W, 2); ERP coordinates
    pers_grid = make_coord(imgHW, flatten=False).to(device) # (h, w, 2)

    for theta, phi in views:  
        ### ERP2PERS ###      
        erp2pers_grid, valid_mask = gridy2x_erp2pers(gridy=erp_grid,
            HWy=erpHW, HWx=imgHW, THETA=theta, PHI=phi, FOVy=360, FOVx=90) # (H*W, 2), (H*W)
    
        # Filter valid indices
        erp_indices = torch.arange(0, erpHW[0]*erpHW[1]).to(device) # (H*W,)
        erp_indices = erp_indices[valid_mask.bool()].long() # sample (D,) from (H*W,)

        erp2pers_grid = erp2pers_grid[valid_mask.bool()] # sample (D, 2) from (H*W, 2)
        erp2pers_grid = erp2pers_grid.unsqueeze(1).unsqueeze(0) # (D, 2) -> (1, D, 1, 2)
        erp2pers_grid = erp2pers_grid.flip(-1) # x <-> y

        erp2pers_pairs.append((erp2pers_grid, erp_indices))

        ### PERS2ERP ###
        pers2erp_grid, valid_mask = gridy2x_pers2erp(gridy=pers_grid,
            HWy=imgHW, HWx=erpHW, THETA=theta, PHI=phi, FOVy=fov, FOVx=360)
        pers2erp_grid = pers2erp_grid.view(*imgHW, 2).unsqueeze(0).flip(-1)
        pers2erp_grids.append(pers2erp_grid)

    return erp2pers_pairs, pers2erp_grids

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
        erp_img.expand(len(pers2erp_grids), -1, -1 ,-1),
        pers2erp_grids_input,
        padding_mode="reflection",
        mode="bilinear", align_corners=False)

    img_js.clamp_(0, 1)

    img_js = img_js.unsqueeze(1)

    return list(torch.unbind(img_js, dim=0))


@torch.no_grad()
def erp2pers_discrete_warping(views, erp_noise, fov, pers_hw): # erp2pers_discrete_warping
    B, C, H, W = erp_noise.shape
    h, w = pers_hw

    device = erp_noise.device 

    erp_noise_flat = erp_noise.reshape(B*C, -1) # (B*C, H*W)
    erp_pixel_grid = make_coord((H, W), flatten=True).to(device) # (H*W, 2)
    pers_pixel_idx = torch.arange(1, h*w+1, dtype=torch.float32).to(device) # (h*w, )
    count_values = torch.ones_like(erp_noise_flat[:1]) # (1, H*W)

    pers_noises = []

    for theta, phi in views:
        # 1) Map each ERP pixel on Pers. image grid
        erp2pers_grid, valid_mask = gridy2x_erp2pers(gridy=erp_pixel_grid,
            HWy=(H, W), HWx=(h, w), THETA=theta, PHI=phi, FOVy=360, FOVx=fov)
        valid_erp2pers_grid = erp2pers_grid[valid_mask.bool()] # (D, 2) for D < H*W
        
        # 2) Find nearest Pers. pixel index of each erp pixel
        valid_erp2pers_idx = F.grid_sample(
            pers_pixel_idx.view(1, 1, h, w),
            valid_erp2pers_grid.view(1, 1, -1, 2).flip(-1),
            mode="nearest", align_corners=False).view(1, -1) # (1, D)
        valid_erp2pers_idx = valid_erp2pers_idx.to(torch.int64)

        # 3) Get warped Pers. noise
        fin_v_val = torch.zeros(B*C, h*w+1, device=device).scatter_add_(
            1, index=valid_erp2pers_idx.repeat(B*C, 1), src=erp_noise_flat)[..., 1:] # (B*C, h*w)
        fin_v_num = torch.zeros(1, h*w+1, device=device).scatter_add_(
            1, index=valid_erp2pers_idx, src=count_values)[..., 1:] # (1, h*w)
        
        assert fin_v_num.min() != 0, ValueError(f"There are some pixels that do not match with any ERP pixel. ({theta},{phi})")
        final_values = fin_v_val / torch.sqrt(fin_v_num)
        pers_noise = final_values.reshape(B, C, h, w)
        pers_noise = pers_noise.to(device)
        pers_noises.append(pers_noise)

    return pers_noises

@torch.no_grad()
def generate_views(panorama_height, panorama_width, window_size=64, stride=(8, 8)):
    if isinstance(stride, int):
        stride = (stride, stride)
    stride_h, stride_w = stride
    num_blocks_height = (panorama_height - window_size) // stride_h + 1
    num_blocks_width = (panorama_width - window_size) // stride_w + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride_h)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride_w)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views