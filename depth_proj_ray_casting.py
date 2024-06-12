import open3d as o3d
from glob import glob
import torch
import frnn
import numpy as np
import imageio
from datasets.data_io import read_cam_file, read_map

def uniform_sampling(num_points, rays_o, rays_d, depth_range):
    B, N_rays, _ = rays_o.shape
    n_samples = num_points
    depth_range = depth_range.squeeze()
    near, far = depth_range[0], depth_range[1]
    near, far = near.reshape(1, 1, 1).repeat(1, N_rays, 1), far.reshape(1, 1, 1).repeat(1, N_rays, 1)
    interval = torch.linspace(0., 1., n_samples, device=rays_o.device)[None, None] #[1, 1, N_samples]
    z_vals = near + (far - near) * interval #[1, N_rays, N_samples]

    dist_value = z_vals[0, 0, 1] - z_vals[0, 0, 0]
    dist = torch.ones_like(z_vals) * dist_value

    pts = rays_o[..., None, :] + rays_d[..., None, :] * (z_vals[..., None])  # [B, N_rays, N_samples, 3]
    rays_d = (rays_d[..., None, :]).expand_as(pts)
    rays_o = (rays_o[..., None, :]).expand_as(pts)

    pts_info = {"world_xyz": pts.reshape(B, -1, 3), "z_vals": z_vals.reshape(B, -1), "rays_o":rays_o.reshape(B, -1, 3), "rays_d": rays_d.reshape(B, -1, 3), "dist":dist.reshape(B, -1), "R":N_rays, "SR":n_samples}
    return pts_info

def lift(x, y, z, intrinsics):

    device = x.device
    # parse intrinsics
    intrinsics = intrinsics.to(device)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    sk = torch.FloatTensor([0]).to(device)
    # print("intrinsics[..., 0, 1]", intrinsics[..., 0, 1])
    if intrinsics[..., 0, 1] != 0:
        sk = intrinsics[..., 0, 1]

    x_lift = (
        (
            x
            - cx.unsqueeze(-1)
            + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
            - sk.unsqueeze(-1) * y / fy.unsqueeze(-1)
        )
        / fx.unsqueeze(-1)
        * z
    )
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(device)), dim=-1)

def build_rays(intrinsics, c2w, H, W, N_rays=-1):

    device = c2w.device
    cam_loc = c2w[..., :3, 3]
    p = c2w

    prefix = c2w.shape[:-2]

    if N_rays > 0:
        N_rays = min(N_rays, H * W)
        i, j = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
        i = i.to(device)
        j = j.to(device)
        coords = torch.stack([i, j], -1).reshape(-1, 2)
        select_inds = np.random.choice(coords.shape[0], size=[N_rays], replace=False)  # (N_rays,)
        coords = coords[select_inds]  # (N_rays, 2)
    else:
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H).to(device), torch.linspace(0, W - 1, W).to(device)), -1)
        coords = coords.reshape(-1, 2) #[H*W, 2]

    coords = coords.reshape([*[1] * len(prefix), -1, 2]).repeat([*prefix, 1, 1]).long() #[B, N_rays, 2]
    u = coords[..., 1]
    v = coords[..., 0]
    pixel_points_cam = lift(u, v, torch.ones_like(u).to(device), intrinsics=intrinsics) #[prefix..., N_rays, 4]
    rays_d = pixel_points_cam[..., :3]
    rays_d = rays_d / torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True)
    rays_d = torch.matmul(p[..., :3, :3].unsqueeze(0), rays_d[..., None]).squeeze(-1) #only rotation without translation

    rays_o = cam_loc[..., None, :].expand_as(rays_d)

    # gt = tgt_img[:, :, coords[..., :, 0], coords[..., :, 1]] #[B, 3, 2048]
    # gt = gt.squeeze().reshape([*[1] * len(prefix), 3, -1]).repeat([*prefix, 1, 1]).permute(0, 2, 1) #[B, 2048, 3]

    return rays_o, rays_d, coords

def depth_gen_by_ray_casting(points, intrinsic, extrinsic, depth_params, H=512, W=640, K=8, r=2.5, sampling_points=128):

    num_points=sampling_points

    points_scaffold = torch.from_numpy(points)[None].cuda().float()
    if depth_params[1] < 3:
        depth_params[1] = 935.0

    c2w = np.linalg.inv(extrinsic)
    #
    intrinsic = torch.from_numpy(intrinsic)[None]
    extrinsic = torch.from_numpy(extrinsic)[None]
    c2w = torch.inverse(extrinsic)
    rays_o, rays_d, _ = build_rays(intrinsic, c2w, H, W)
    rays_o, rays_d = rays_o.cuda(), rays_d.cuda()
    depth_params = torch.from_numpy(depth_params).cuda()
    pts_info = uniform_sampling(num_points, rays_o, rays_d, depth_params)
    z_vals = pts_info["z_vals"].reshape(1, 327680, num_points)

    sampled_pts = pts_info["world_xyz"]
    dist, indices, _, _ = frnn.frnn_grid_points(sampled_pts,
                                                points_scaffold,
                                                K=K,
                                                r=r,
                                                return_nn=False,
                                                return_sorted=True)  # (1, P1, K)
    dist = dist.reshape(1, 327680, num_points, -1)
    dist[dist == -1] = 1e7
    density_func = lambda x: torch.sum(torch.exp(-torch.pow(dist, 2)/2), dim=-1) / K
    density = density_func(dist)
    # density = density / (torch.sum(density, dim=-1, keepdim=True)+1e-5)
    T = -torch.cumsum(density * 10 *r, dim=-1)[..., :-1]
    T = torch.exp(T)
    T = torch.cat([torch.ones_like(T[..., 0:1]), T], dim=-1)
    weights = density * T
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-10)
    value, index = torch.max(weights, dim=-1)
    index = index.squeeze()
    # print(weights.max())
    # print(weights.min())
    # z_vals = z_vals[0, 0, :]
    # depth_map = z_vals[index].reshape(512, 640)

    depth_map = torch.sum(weights * z_vals, dim=-1).reshape(512, 640)

    return depth_map


