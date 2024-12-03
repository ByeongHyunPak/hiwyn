import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr

from geometry import gridy2x_pers2erp

def cond_noise_sampling(src_noise, level=3):

    B, C, H, W = src_noise.shape

    up_factor = 2 ** level

    upscaled_means = F.interpolate(src_noise, scale_factor=(up_factor, up_factor), mode='nearest')

    up_H = up_factor * H
    up_W = up_factor * W

    """
        1) Unconditionally sample a discrete Nk x Nk Gaussian sample
    """

    raw_rand = torch.randn(B, C, up_H, up_W, device=src_noise.device)

    """
        2) Remove its mean from it
    """

    Z_mean = raw_rand.unfold(2, up_factor, up_factor).unfold(3, up_factor, up_factor).mean((4, 5))
    Z_mean = F.interpolate(Z_mean, scale_factor=up_factor, mode='nearest')
    mean_removed_rand = raw_rand - Z_mean

    """
        3) Add the pixel value to it
    """

    up_noise = upscaled_means / up_factor + mean_removed_rand

    return up_noise

def erp2pers_latent_warping(erp_noise, HW_pers, views, glctx):
    device = erp_noise.device
    B, C, H_erp, W_erp = erp_noise.shape
    H_pers, W_pers = HW_pers

    # Defining the partitioned polygons for target noise map
    tr_H_pers, tr_W_pers = 2 * H_pers + 1, 2 * W_pers + 1
    i, j = torch.meshgrid(
        torch.arange(tr_H_pers, dtype=torch.int32),
        torch.arange(tr_W_pers, dtype=torch.int32),
        indexing="ij")
    mesh_idxs = torch.stack((i, j), dim=-1).to(device)
    reshaped_mesh_idxs = mesh_idxs.reshape(-1,2)
    
    front_tri_verts = torch.tensor([
            [0, 1, 1+tr_W_pers], 
            [0, tr_W_pers, 1+tr_W_pers], 
            [tr_W_pers, 1+tr_W_pers, 1+2*tr_W_pers], 
            [tr_W_pers, 2*tr_W_pers, 1+2*tr_W_pers]], 
            device=device)
    per_tri_verts = torch.cat((front_tri_verts, front_tri_verts + 1), dim=0)
    width = torch.arange(0, tr_W_pers - 1, 2)
    height = torch.arange(0, tr_H_pers-1, 2) * tr_W_pers
    start_idxs = (width[None,...] + height[...,None]).reshape(-1,1).to(device)
    vertices = (start_idxs.repeat(1, 8)[..., None] + per_tri_verts[None, ...]).reshape(-1, 3)
    
    # Perspective view vertex grid
    pers_i, pers_j = torch.meshgrid(
        torch.linspace(-1, 1, tr_H_pers, dtype=torch.float64),
        torch.linspace(-1, 1, tr_W_pers, dtype=torch.float64),
        indexing="ij")
    pers_grid = torch.stack((pers_i, pers_j), dim=-1).to(device)

    res = []
    inds = []

    for theta, phi in views:
        # Warping Rasterized Pers. grid
        pers2erp_grid, _ = gridy2x_pers2erp(gridy=pers_grid,
            HWy=(2*H_pers, 2*W_pers), HWx=(2*H_erp, 2*W_erp),
            THETA=theta, PHI=phi, FOVy=90, FOVx=360)
        
        pers_to_erp_map = pers2erp_grid.view(tr_H_pers, tr_W_pers, 2)
        idx_y = reshaped_mesh_idxs[..., 0].int()
        idx_x = reshaped_mesh_idxs[..., 1].int()
        warped_coords = pers_to_erp_map[idx_y, idx_x].fliplr()

        len_grid = idx_y.shape[0]
        zeros = torch.zeros(len_grid, 1).to(device)
        ones = torch.ones(len_grid, 1).to(device)   
        warped_vtx_pos = torch.cat((warped_coords, zeros, ones), dim=-1)
        warped_vtx_pos = warped_vtx_pos[None,...].to("cuda")
        vertices = vertices.int().to("cuda")

        resolution = [H_erp, W_erp]
        with torch.no_grad():
            rast_out, _ = dr.rasterize(glctx, warped_vtx_pos, vertices, resolution=resolution)
        rast = rast_out[:,:,:,3:].permute(0,3,1,2).to(torch.int64)

        # Finding pixel indices in cond-upsampled map
        indices = (rast - 1) // 8 + 1 # there is 8 triangles per pixel
        erp_up_noise_flat = erp_noise.reshape(B*C, -1)
        ones_flat = torch.ones_like(erp_up_noise_flat[:1])
        indices_flat = indices.reshape(1, -1).to(torch.int64)

        # Get warped target noise
        fin_v_val = torch.zeros(B*C, H_pers*W_pers+1, device=device)\
            .scatter_add_(1, index=indices_flat.repeat(B*C, 1), src=erp_up_noise_flat)[..., 1:]
        fin_v_num = torch.zeros(1, H_pers*W_pers+1, device=device)\
            .scatter_add_(1, index=indices_flat, src=ones_flat)[..., 1:]
        assert fin_v_num.min() != 0, ValueError(f"{theta},{phi}")

        final_values = fin_v_val / torch.sqrt(fin_v_num)
        pers_warped_noise = final_values.reshape(B, C, H_pers, W_pers).float()
        pers_warped_noise = pers_warped_noise.to(device)

        res.append(pers_warped_noise)
        inds.append(indices.reshape(*resolution))
        fin_v_num = fin_v_num.reshape(1, 1, H_pers, W_pers)

    return res, inds, fin_v_num

def compute_erp_up_noise_pred(pers_noise_pred, erp2pers_ind, fin_v_num):

    device = pers_noise_pred.device
    B, C, H_pers, W_pers = pers_noise_pred.shape
    H_erp_up, W_erp_up = erp2pers_ind.shape

    # Initialize result tensors in smaller chunks
    erp_up_noise_pred = torch.zeros(B*C, H_erp_up*W_erp_up, device=device)
    
    # Normalize perspective noise
    pers_noise_pred = pers_noise_pred * torch.sqrt(fin_v_num)
    pers_noise_pred_flat = pers_noise_pred.reshape(B*C, -1)

    # Avoid creating a padded tensor unnecessarily
    pers_noise_pred_flat_pad = torch.zeros(B*C, H_pers*W_pers + 1, device=device)
    pers_noise_pred_flat_pad[:, 1:] = pers_noise_pred_flat  # Add padding in-place

    # Flatten and mask indices without repeating
    erp2pers_ind_flat = erp2pers_ind.flatten()  # (H_erp_up * W_erp_up,)
    valid_mask = erp2pers_ind_flat > 0

    # Incremental updates for noise
    erp_up_noise_pred[:, ...] = pers_noise_pred_flat_pad[:, erp2pers_ind_flat]
    erp_up_noise_pred = erp_up_noise_pred.reshape(B, C, H_erp_up, W_erp_up)
    valid_mask = valid_mask.reshape(1, 1, H_erp_up, W_erp_up)

    return erp_up_noise_pred, valid_mask



# # deprecated -> do no average noise_preds
# def compute_erp_up_noise_pred(pers_noise_preds, erp2pers_indices, fin_v_num):

#     device = pers_noise_preds[0].device
#     B, C, H_pers, W_pers = pers_noise_preds[0].shape
#     H_erp_up, W_erp_up = erp2pers_indices[0].shape

#     # Initialize result tensors in smaller chunks
#     erp_up_noise_pred = torch.zeros(B*C, H_erp_up*W_erp_up, device=device)
#     erp_up_count = torch.zeros(B*C, H_erp_up*W_erp_up, device=device)

#     # Process batch-wise to avoid large memory allocation
#     for i, (pers_noise_pred, erp2pers_ind) in enumerate(zip(pers_noise_preds, erp2pers_indices)):

#         # # Normalize perspective noise
#         pers_noise_pred = pers_noise_pred / torch.sqrt(fin_v_num)
#         pers_noise_pred_flat = pers_noise_pred.reshape(B*C, -1)

#         # Avoid creating a padded tensor unnecessarily
#         pers_noise_pred_flat_pad = torch.zeros(B*C, H_pers*W_pers + 1, device=device)
#         pers_noise_pred_flat_pad[:, 1:] = pers_noise_pred_flat  # Add padding in-place

#         # Flatten and mask indices without repeating
#         erp2pers_ind_flat = erp2pers_ind.flatten()  # (H_erp_up * W_erp_up,)
#         valid_mask = erp2pers_ind_flat > 0
#         # valid_indices = erp2pers_ind_flat[valid_mask]  # Only valid indices

#         # Incremental updates for noise and count
#         for b in range(B*C):
#             erp_up_noise_pred[b] += pers_noise_pred_flat_pad[b, erp2pers_ind_flat]
#             erp_up_count[b, valid_mask] += 1

#     # Avoid division by zero
#     erp_up_count = torch.clamp(erp_up_count, min=1)

#     # Compute averaged noise_pred
#     erp_up_noise_pred = erp_up_noise_pred / erp_up_count
#     erp_up_noise_pred = erp_up_noise_pred.reshape(B, C, H_erp_up, W_erp_up)

#     return erp_up_noise_pred
