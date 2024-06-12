import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import os

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


def build_rays(intrinsics, c2w, tgt_img, mask, N_rays=-1, sample_on_mask=None):

    device = c2w.device
    H, W = tgt_img.shape[-2:]
    cam_loc = c2w[..., :3, 3]
    p = c2w

    prefix = c2w.shape[:-2]

    if N_rays > 0:
        if sample_on_mask:
            #mask [B(1), 1, H, W]
            mask = mask.squeeze() #[H, W]
            mask_sum = int(mask.sum().int().cpu().data.numpy())
            N_rays = min(mask_sum, N_rays)
            pixel_valid = mask.nonzero()
            permutation = np.random.permutation(mask_sum)[:N_rays].astype(np.int32)
            X, Y = pixel_valid[:,1][permutation], pixel_valid[:,0][permutation]
            coords = torch.stack([Y,X], dim=-1) #[N_rays, 2]
        else:
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

    gt = tgt_img[:, :, coords[..., :, 0], coords[..., :, 1]] #[B, 3, 2048]
    gt = gt.squeeze().reshape([*[1] * len(prefix), 3, -1]).repeat([*prefix, 1, 1]).permute(0, 2, 1) #[B, 2048, 3]

    return rays_o, rays_d, gt, coords

def log_sampling(args, rays_o, rays_d, depth, selected_inds, depth_range, noise=False):
    B, N_rays, _ = rays_o.shape
    base = args.sample_strategy.base
    if noise:
        base = base + torch.normal(0,0.0075,size=(1,)).item()
    n_samples = args.render.N_Samples
    center_flag = False
    if n_samples % 2 == 1:
        n_samples -= 1
        center_flag = True
    half_n = n_samples // 2
    pos_increment = torch.logspace(0, half_n - 1, half_n, base=base).to(rays_d.device)
    neg_increment = -torch.flip(pos_increment, dims=[0])
    if center_flag:
        vals = torch.cat([neg_increment,  torch.FloatTensor([0.]).to(rays_o.device),  pos_increment], dim=0) #[n_samples]
    else:
        vals = torch.cat([neg_increment,  pos_increment], dim=0) #[n_samples]
    vals = vals.view(1, 1, -1)
    depth_range = depth_range.squeeze()

    near_bounds, far_bounds = depth_range[0], depth_range[1]
    depth, selected_inds = depth.squeeze(), selected_inds.squeeze()
    depth = depth[selected_inds[:, 0],selected_inds[:, 1]]
    depth = depth.view(1, -1, 1)
    depth[depth < near_bounds] = near_bounds
    depth[depth > far_bounds] = far_bounds
    if noise:
        uni_tensor = (10 * torch.rand(size=(1,depth.shape[1],1)) - 5).to(depth.device)
        depth = depth + uni_tensor

    z_vals = depth + vals

    dist = z_vals[0, 0, 1:] - z_vals[0, 0, :-1]
    dist = dist[None, None]
    dist = torch.cat([dist, torch.ones_like(dist[:,:,0:1])*1e2],dim=-1)
    dist = dist.repeat(B, N_rays, 1)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * (z_vals[..., None]) #[B, N_rays, N_samples, 3]

    rays_d = (rays_d[..., None, :]).expand_as(pts)
    rays_o = (rays_o[..., None, :]).expand_as(pts)

    pts_info = {"world_xyz":pts.reshape(B, -1, 3), "z_vals":z_vals.reshape(B, -1), "rays_o":rays_o.reshape(B, -1, 3), "rays_d":rays_d.reshape(B, -1, 3), "dist":dist.reshape(B, -1), "R":N_rays, "SR":n_samples + 1 if center_flag else n_samples}

    return pts_info

def surface_sampling(args, rays_o, rays_d, depth, selected_inds, depth_range):
    B, N_rays, _ = rays_o.shape
    interval = args.render.depth_interval
    n_samples = args.render.N_Samples

    depth_range = depth_range.squeeze()

    near_bounds, far_bounds = depth_range[0], depth_range[1]
    depth, selected_inds = depth.squeeze(), selected_inds.squeeze()
    depth = depth[selected_inds[:, 0],selected_inds[:, 1]]

    depth[depth < near_bounds] = near_bounds
    depth[depth > far_bounds] = far_bounds
    ray_near = depth - interval / 2
    ray_far = depth + interval / 2

    ray_near, ray_far = ray_near.reshape(1, -1, 1), ray_far.reshape(1, -1, 1)
    if n_samples == 1:
        z_vals = ray_near + (ray_far - ray_near) * 0.5

    else:
        z_vals = ray_near + (ray_far - ray_near) * (torch.linspace(0., 1., n_samples, device=depth.device)[None, None])

    dist_value = z_vals[0, 0, 1] - z_vals[0, 0, 0]
    dist = torch.ones_like(z_vals) * dist_value

    pts = rays_o[..., None, :] + rays_d[..., None, :] * (z_vals[..., None]) #[B, N_rays, N_samples, 3]

    rays_d = (rays_d[..., None, :]).expand_as(pts)
    rays_o = (rays_o[..., None, :]).expand_as(pts)

    pts_info = {"world_xyz":pts.reshape(B, -1, 3), "z_vals":z_vals.reshape(B, -1), "rays_o":rays_o.reshape(B, -1, 3), "rays_d":rays_d.reshape(B, -1, 3), "dist":dist.reshape(B, -1), "R":N_rays, "SR":n_samples}

    return pts_info

def uniform_sampling(args, rays_o, rays_d, depth_range):
    B, N_rays, _ = rays_o.shape
    n_samples = args.render.uniform_samples
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

def depth_projection(points, intrinsics, extrinsics):
    # points [1, N, 3] int [1,3,3] ext [1,4,4]
    points = torch.cat([points, torch.ones_like(points)], dim=-1)
    points_cam = points @ extrinsics.transpose(-1, -2)[..., :3]
    points_pixel = points_cam[..., :3] @ intrinsics.transpose(-1, -2)
    uv = points_pixel[...,:2] / torch.clamp_min(points_pixel[..., 2:3], 1e-7)
    uv_depth = torch.cat([uv, points_pixel[..., 2:3]], dim=-1)

    return uv_depth

def visibility_score(points, intrinsics, extrinsics, depth_gt):
    points = torch.cat([points, torch.ones_like(points[..., 0:1])], dim=-1)
    # print("points shape ", points.shape)
    # print("ext shape ", extrinsics.transpose(-1, -2).shape)

    points_cam = (points @ extrinsics.transpose(-1, -2))[..., :3]
    points_pixel = points_cam[..., :3] @ intrinsics.transpose(-1, -2)
    uv = points_pixel[..., :2] / torch.clamp_min(points_pixel[..., 2:3], 1e-6) #[B, N, 2]
    depth = points_pixel[..., 2:3]

    H, W = depth_gt.shape[-2:] #[1, 1, H, W]
    uv[..., 0], uv[..., 1] = uv[..., 0] / (W - 1), uv[..., 1] / (H - 1)
    uv = uv * 2 - 1
    interpolated_depth = F.grid_sample(depth_gt, grid=uv[:, None], align_corners=True, mode="bilinear", padding_mode="zeros").permute(0, 2, 3, 1)[:, 0] #[B, N, 1]
    score = torch.abs(depth - interpolated_depth) / depth
    score = 1 - score
    # score = 1 / (score + 1e-6) #[B, N, 1]

    return score

def batchify_neural_points_query(query_fn, sampled_points_info, chunk=512):
    # batchify rays into small minibatch
    N_pts = sampled_points_info["world_xyz"].shape[1]
    color_list = []
    sigma_list = []
    for i in range(0, N_pts, chunk):
        temp_pts_info = {}
        temp_pts_info["weights"] = sampled_points_info["weights"][:, i:i + chunk]
        temp_pts_info["indices"] = sampled_points_info["indices"][:, i:i + chunk]
        temp_pts_info["world_xyz"] = sampled_points_info["world_xyz"][:, i:i + chunk]
        temp_pts_info["rays_d"] = sampled_points_info["rays_d"][:, i:i + chunk]
        # temp_pts_info["rays_o"] = sampled_points_info["rays_o"][:, i:i + chunk]
        # temp_pts_info["depth"] = sampled_points_info["depth"][..., i:i + chunk]
        temp_pts_info["extrinsics"] = sampled_points_info["extrinsics"]
        # temp_pts_info["dist"] = sampled_points_info["dist"]

        color, sdf = query_fn(temp_pts_info)
        color_list.append(color)
        sigma_list.append(sdf)
    # N_pts = sampled_points_info["alive_points"].shape[1]
    # color_list = []
    # sigma_list = []
    # for i in range(0, N_pts, chunk):
    #     temp_pts_info = {}
    #     temp_pts_info["weights"] = sampled_points_info["weights"][:, i:i + chunk]
    #     temp_pts_info["indices"] = sampled_points_info["indices"][:, i:i + chunk]
    #     temp_pts_info["alive_points"] = sampled_points_info["alive_points"][:, i:i + chunk]
    #     temp_pts_info["rays_d"] = sampled_points_info["rays_d"][:, i:i + chunk]
    #     temp_pts_info["extrinsics"] = sampled_points_info["extrinsics"]
    #
    #     color, sigma = query_fn(temp_pts_info)
    #     color_list.append(color)
    #     sigma_list.append(sigma)

    return torch.cat(color_list, dim=1), torch.cat(sigma_list, dim=1)

def reproject_with_depth(
    args,
    img_ref,
    depth_ref,
    intrinsics_ref,
    c2w,
    # depth_src: np.ndarray,
    # intrinsics_src: np.ndarray,
    # extrinsics_src: np.ndarray
):
    """
    :param depth_ref: 1,1,H,W
    :param intrinsics_ref: 1,3,3
    :param extrinsics_ref: 1,4,4
    :return:
    """
    img_ref = img_ref.squeeze()
    intrinsics_ref = intrinsics_ref.squeeze()
    c2w = c2w.squeeze()


    print("inv intr in repo")
    print(torch.linalg.inv(intrinsics_ref))
    print("c2w in repro")
    print(c2w)

    device = args.device
    W, H = depth_ref.shape[-1], depth_ref.shape[-2]
    # mask =
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H).to(device), torch.linspace(0, W - 1, W).to(device)),-1)
    coords = coords.reshape(-1, 2)
    coords = torch.cat([coords, torch.ones_like(coords[:, 0:1]).to(device)],dim=-1)
    u = coords[..., 1:2]
    v = coords[..., 0:1]
    xyz = torch.cat([u, v, torch.ones_like(u).to(device)], dim=-1)
    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), xyz.transpose(1, 0) * depth_ref.reshape(-1))
    xyz_ref = torch.cat([xyz_ref, torch.ones_like(xyz_ref[0:1, :]).to(device)],dim=0)
    world_xyz = torch.matmul(c2w, xyz_ref)[:3].transpose(1, 0)

    test_dir = os.path.join("log", args.expname, "test")
    pcd = o3d.geometry.PointCloud()
    points = world_xyz.reshape(-1, 3).cpu().detach().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = img_ref.reshape(-1, 3).cpu().detach().numpy()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_name = "point_cloud_in_train.ply"
    pcd_name = os.path.join(test_dir, pcd_name)
    o3d.io.write_point_cloud(pcd_name, pcd)

    pts_info = {"world_xyz":world_xyz.reshape(1, -1, 1, 3), "z_vals":depth_ref.reshape(1, -1, 1)}

    return pts_info

def cdf_Phi_s(x, s):  # \VarPhi_s(t)
    # den = 1 + torch.exp(-s*x)
    # y = 1./den
    # return y
    return torch.sigmoid(x * s)

def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    sdf = torch.cat([sdf, torch.ones_like(sdf[..., 0:1])], dim=-1)
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha

def sdf_to_sigma(sdf: torch.Tensor, gamma):
    numerator = torch.exp(- (sdf / gamma))
    denominator = gamma * ((1 + torch.exp(- (sdf/gamma)))**2)

    return numerator/denominator

