import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import sys
import pandas as pd
import open3d as o3d
import numpy as np

def pc_world_to_pc_cam(pc, extrinsic):
    pc = np.concatenate([pc, np.ones_like(pc[..., 0:1])], axis=-1)
    pc = pc @ extrinsic.T
    return pc

def world2cam_tensor(pc, extrinsic):
    pc = torch.cat([pc, torch.ones_like(pc[..., 0:1]).to(pc.device)], dim=-1)
    pc = pc @ extrinsic.T
    return pc


Trans = np.array([[1., 0, 0],
                     [0, -1, 0],
                     [0, 0, -1]])


def points2depth(pc_world, H, W, intrinsics, extrinsics):
    pc = pc_world_to_pc_cam(pc_world, extrinsics)[..., :3]  # (134547, 3)
    nerf_flag=False
    if nerf_flag:
        pc = pc @ Trans
    z = pc[..., 2]
    # CAM_WID, CAM_HGT = W, H  #
    # CAM_FX, CAM_FY = intrinsics[0, 0], intrinsics[1, 1]  # fx/fy
    # CAM_CX, CAM_CY = 0.5 * W, 0.5 * H  # cx/cy
    # u = np.round(pc[:, 0] * CAM_FX / z + CAM_CX).astype(int)  # (134547,)
    # v = np.round(pc[:, 1] * CAM_FY / z + CAM_CY).astype(int)  # (134547,)
    point_i = (pc @ intrinsics.T)
    point_i = point_i[..., :2] / point_i[..., 2:]
    u, v = np.round(point_i[..., 0]).astype(np.int), np.round(point_i[..., 1]).astype(np.int)

    valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < W)), np.bitwise_and((v >= 0), (v < H)))

    u, v, z = u[valid], v[valid], z[valid]
    img_z = np.full((H, W), 10000, dtype=np.float32)
    for ui, vi, zi in zip(u, v, z):
        img_z[vi, ui] = min(img_z[vi, ui], zi)

    img_z_shift = np.array([img_z, np.roll(img_z, 1, axis=0), \
                            np.roll(img_z, -1, axis=0), \
                            np.roll(img_z, 1, axis=1), \
                            np.roll(img_z, -1, axis=1)])
    img_z = np.min(img_z_shift, axis=0)
    img_z[img_z == 10000] = 0

    return img_z

def points2depth_ras(pc_world, H, W, intrinsics, extrinsics, nerf_flag=False):
    pass

def points2depth_tensor(pc_world, H, W, intrinsics, extrinsics):
    pc = world2cam_tensor(pc_world, extrinsics)[..., :3]  # (134547, 3)
    nerf_flag=False
    if nerf_flag:
        pc = pc @ Trans
    z = pc[..., 2]

    point_i = (pc @ intrinsics.T)
    point_i = point_i[..., :2] / point_i[..., 2:]
    u, v = torch.round(point_i[..., 0]).int(), torch.round(point_i[..., 1]).int()

    valid = torch.bitwise_and(torch.bitwise_and((u >= 0), (u < W)), torch.bitwise_and((v >= 0), (v < H)))

    u, v, z = u[valid], v[valid], z[valid]
    img_z = torch.full((H, W), 10000.0)
    for ui, vi, zi in zip(u, v, z):
        img_z[vi, ui] = min(img_z[vi, ui], zi)

    img_z_shift = torch.stack([img_z, torch.roll(img_z, 1, dims=0), \
                            torch.roll(img_z, -1, dims=0), \
                            torch.roll(img_z, 1, dims=1), \
                            torch.roll(img_z, -1, dims=1)], dim=0)
    
    img_z = torch.min(img_z_shift, dim=0, keepdim=False)[0]
    img_z[img_z == 10000] = 0

    return img_z
