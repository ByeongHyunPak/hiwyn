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

    raw_rand = torch.randn(B, C, up_H, up_W)

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

def erp2pers_noise_warping(src_noise, up_level, HW_src, HW_tgt, views, glctx):

    B, C = src_noise.shape[:2]
    H_src, W_src = HW_src
    H_tgt, W_tgt = HW_tgt
    
    if up_level == 1:
        src_up_noise = src_noise
    elif up_level > 1:
        src_up_noise = cond_noise_sampling(src_noise, up_level)
    else:
        NotImplementedError
    
    # Defining the partitioned polygons for target noise map
    tr_H_tgt, tr_W_tgt = 2 * H_tgt + 1, 2 * W_tgt + 1
    i, j = torch.meshgrid(
        torch.arange(tr_H_tgt, dtype=torch.int32),
        torch.arange(tr_W_tgt, dtype=torch.int32),
        indexing="ij")
    mesh_idxs = torch.stack((i, j), dim=-1)
    reshaped_mesh_idxs = mesh_idxs.reshape(-1,2)
    
    front_tri_verts = torch.tensor([
        [0, 1, 1+tr_W_tgt], [0, tr_W_tgt, 1+tr_W_tgt], 
        [tr_W_tgt, 1+tr_W_tgt, 1+2*tr_W_tgt], [tr_W_tgt, 2*tr_W_tgt, 1+2*tr_W_tgt]])
    per_tri_verts = torch.cat((front_tri_verts, front_tri_verts + 1),dim=0)
    width = torch.arange(0, tr_W_tgt - 1, 2)
    height = torch.arange(0, tr_H_tgt-1, 2) * (tr_W_tgt)
    start_idxs = (width[None,...] + height[...,None]).reshape(-1,1)
    vertices = (start_idxs.repeat(1,8)[...,None] + per_tri_verts[None,...]).reshape(-1,3)
    
    # Perspective view vertex grid
    pers_i, pers_j = torch.meshgrid(
        torch.linspace(-1, 1, tr_H_tgt),
        torch.linspace(-1, 1, tr_W_tgt),
        indexing="ij")
    pers_grid = torch.stack((pers_i, pers_j), dim=-1)

    res = []
    inds = []
    for theta, phi in views:
        # Warping Rasterized Pers. grid
        pers2erp_grid, _ = gridy2x_pers2erp(gridy=pers_grid,
            HWy=(2*H_tgt, 2*W_tgt), HWx=(2*H_src, 2*W_src),
            THETA=theta, PHI=phi, FOVy=90, FOVx=360)
        
        tgt_to_src_map = pers2erp_grid.view(tr_H_tgt, tr_W_tgt, 2)
        idx_y = reshaped_mesh_idxs[..., 0].int()
        idx_x = reshaped_mesh_idxs[..., 1].int()
        warped_coords = tgt_to_src_map[idx_y, idx_x].fliplr()

        len_grid = idx_y.shape[0]
        warped_vtx_pos = torch.cat((warped_coords, torch.zeros(len_grid, 1), torch.ones(len_grid, 1)), dim=-1)
        warped_vtx_pos = warped_vtx_pos[None,...].to("cuda")
        vertices = vertices.int().to("cuda")

        resolution = [H_src * (2 ** up_level), W_src * (2 ** up_level)]
        with torch.no_grad():
            rast_out, _ = dr.rasterize(glctx, warped_vtx_pos, vertices, resolution=resolution)
        rast = rast_out[:,:,:,3:].permute(0,3,1,2).to(torch.int64)

        # Finding pixel indices in cond-upsampled map
        indices = (rast - 1) // 8 + 1 # there is 8 triangles per pixel
        src_up_noise_flat = src_up_noise.reshape(B*C, -1).cpu()
        ones_flat = torch.ones_like(src_up_noise_flat[:1])
        indices_flat = indices.reshape(1, -1).cpu().to(torch.int64)

        # Get warped target noise
        fin_v_val = torch.zeros(B*C, H_tgt*W_tgt+1).scatter_add_(1, index=indices_flat.repeat(B*C, 1), src=src_up_noise_flat)[..., 1:]
        fin_v_num = torch.zeros(1, H_tgt*W_tgt+1).scatter_add_(1, index=indices_flat, src=ones_flat)[..., 1:]
        assert fin_v_num.min() != 0, ValueError(f"{theta},{phi}")

        final_values = fin_v_val / torch.sqrt(fin_v_num)
        tgt_warped_noise = final_values.reshape(B, C, H_tgt, W_tgt).float()
        tgt_warped_noise = tgt_warped_noise.cuda()
        res.append(tgt_warped_noise)
        inds.append(indices.reshape(*resolution).to(torch.int64))
        fin_v_num = fin_v_num.reshape(1, 1, H_tgt, W_tgt).cuda()

    return res, inds, src_up_noise, fin_v_num