@torch.no_grad()
def discrete_warping(self, src_white_noise, normalize_fin_v_val=True):
    B, C, H, W = src_white_noise.shape
    h, w = 512//8, 512//8

    # 1) Rasterize target noise map
    tr_h, tr_w = 2*h+1, 2*w+1
    i, j = torch.meshgrid(
        torch.arange(tr_h, dtype=torch.int32),
        torch.arange(tr_w, dtype=torch.int32),
        indexing="ij")
    mesh_idxs = torch.stack((i, j), dim=-1).to(self.device)
    reshaped_mesh_idxs = mesh_idxs.reshape(-1,2)

    front_tri_verts = torch.tensor([
            [0, 1, 1+tr_w], 
            [0, tr_w, 1+tr_w], 
            [tr_w, 1+tr_w, 1+2*tr_w], 
            [tr_w, 2*tr_w, 1+2*tr_w]], 
            device=self.device)
    per_tri_verts = torch.cat((front_tri_verts, front_tri_verts + 1), dim=0)
    width = torch.arange(0, tr_w - 1, 2)
    height = torch.arange(0, tr_h-1, 2) * tr_w
    start_idxs = (width[None,...] + height[...,None]).reshape(-1,1).to(self.device)
    vertices = (start_idxs.repeat(1, 8)[..., None] + per_tri_verts[None, ...]).reshape(-1, 3)

    # 2) Get normalized target's vertex grid
    tgt_i, tgt_j = torch.meshgrid(
        torch.linspace(-1, 1, tr_h, dtype=torch.float64),
        torch.linspace(-1, 1, tr_w, dtype=torch.float64),
        indexing="ij")
    tgt_vtx_grid = torch.stack((tgt_i, tgt_j), dim=-1).to(self.device)

    tgts = []
    indices = []
    fin_v_nums = []

    for theta, phi in self.views:
        # 3) warp tgt_vtx_grid onto source grid
        pers2erp_grid, _ = gridy2x_pers2erp(gridy=tgt_vtx_grid,
            HWy=(2*h, 2*w), HWx=(2*H, 2*W), THETA=theta, PHI=phi, FOVy=90, FOVx=360)
        
        pers_to_erp_map = pers2erp_grid.view(tr_h, tr_w, 2)
        idx_y = reshaped_mesh_idxs[..., 0].int()
        idx_x = reshaped_mesh_idxs[..., 1].int()
        warped_coords = pers_to_erp_map[idx_y, idx_x].fliplr()

        len_grid = idx_y.shape[0]
        zeros = torch.zeros(len_grid, 1).to(self.device)
        ones = torch.ones(len_grid, 1).to(self.device)   
        warped_vtx_pos = torch.cat((warped_coords, zeros, ones), dim=-1)
        warped_vtx_pos = warped_vtx_pos[None,...].to("cuda")
        vertices = vertices.int().to("cuda")

        resolution = [H, W]
        with torch.no_grad():
            rast_out, _ = dr.rasterize(glctx, warped_vtx_pos, vertices, resolution=resolution)
        rast = rast_out[:,:,:,3:].permute(0,3,1,2).to(torch.int64)

        # 4) Finding pixel indices in src_white_noise
        ind = (rast - 1) // 8 + 1 # there is 8 triangles per pixel
        erp_up_noise_flat = src_white_noise.reshape(B*C, -1)
        ones_flat = torch.ones_like(erp_up_noise_flat[:1])
        ind_flat = ind.reshape(1, -1).to(torch.int64)

        # 5) Get warped target noise
        fin_v_val = torch.zeros(B*C, h*w+1, device=self.device) \
            .scatter_add_(1, index=ind_flat.repeat(B*C, 1), src=erp_up_noise_flat)[..., 1:]
        fin_v_num = torch.zeros(1, h*w+1, device=self.device) \
            .scatter_add_(1, index=ind_flat, src=ones_flat)[..., 1:]
        assert fin_v_num.min() != 0, ValueError(f"{theta},{phi}")

        if normalize_fin_v_val:
            final_values = fin_v_val / torch.sqrt(fin_v_num)
        else:
            final_values = fin_v_val
        tgt_warped_noise = final_values.reshape(B, C, h, w).float()
        tgt_warped_noise = tgt_warped_noise.to(self.device)
        
        tgts.append(tgt_warped_noise)
        indices.append(ind.reshape(*resolution))

        if len(self.fin_v_nums) < len(self.views):
            self.fin_v_nums.append(fin_v_num.reshape(h, w))
    
    self.indices = indices
    return tgts

inp = F.grid_sample(hr_pers_img.unsqueeze(0),
                        gridy2x.unsqueeze(0).flip(-1),
                        mode='bicubic',
                        padding_mode='reflection',
                        align_corners=False).clamp_(0, 1)[0]