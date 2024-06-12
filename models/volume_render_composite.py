import taichi as ti
from taichi.math import vec3
import torch
ti.init(arch=ti.cuda)
@ti.kernel
def composite(
           sigmas: ti.types.ndarray(),
             rgbs: ti.types.ndarray(),
        start_indices: ti.types.ndarray(),
        steps: ti.types.ndarray(),
             dist: ti.types.ndarray(),
            z_val: ti.types.ndarray(),
    alive_ray_index: ti.types.ndarray(),
          opacity: ti.types.ndarray(),
            depth: ti.types.ndarray(),
              rgb: ti.types.ndarray()
):
    ti.loop_config(block_dim=256)
    for n in alive_ray_index:
        ray_idx = alive_ray_index[n]
        start_ind = start_indices[n]
        step = steps[n]

        T = 1 - opacity[ray_idx]
        rgb_temp = vec3(0.0)
        depth_temp = 0.0
        opacity_temp = 0.0
        for i in range(step):
            s = start_ind + i
            delta = dist[s]
            a = 1.0 - ti.exp(-sigmas[s]*delta)
            w = a * T
            z = z_val[s]
            rgb_vec3 = vec3(rgbs[s, 0], rgbs[s, 1], rgbs[s, 2])
            rgb_temp += w * rgb_vec3
            depth_temp += w * z
            opacity_temp += w
            T *= 1.0 - a

        rgb[ray_idx, 0] += rgb_temp[0]
        rgb[ray_idx, 1] += rgb_temp[1]
        rgb[ray_idx, 2] += rgb_temp[2]
        depth[ray_idx] += depth_temp
        opacity[ray_idx] += opacity_temp

def composite_torch(sigmas,
             rgbs,
             dist,
            z_val,
    alive_indices,
    alive_ray_index,
          opacity,
            depth,
              rgb):

    for n in range(alive_ray_index.shape[0]):
        ray_idx = alive_ray_index[n]
        mask = (alive_indices == ray_idx).flatten()
        cur_sigma = sigmas[:, mask]
        cur_rgbs = rgbs[:, mask]
        cur_dist = dist[:, mask]
        cur_zval = z_val[:, mask]
        steps = cur_sigma.shape[1]
        T = 1
        rgb_temp = torch.FloatTensor([0.0,0.0,0.0]).to(mask.device)
        depth_temp = 0.0
        opacity_temp = 0.0
        for s in range(steps):

            delta = cur_dist[:, s]

            a = 1.0 - torch.exp(-cur_sigma[:, s]*delta)
            w = a * T

            z = cur_zval[: , s]
            rgb_vec3 = torch.FloatTensor([cur_rgbs[:, s, 0], cur_rgbs[:, s, 1], cur_rgbs[:, s, 2]]).to(mask.device)
            rgb_temp = rgb_temp + (w * rgb_vec3).view(-1)
            # print(" w ", w.reshape(-1))
            # print(cur_rgbs[:, s, 0].shape)
            rgb[ray_idx, 0] = rgb[ray_idx, 0] + w * cur_rgbs[:, s, 0]
            rgb[ray_idx, 1] = rgb[ray_idx, 1] + w * cur_rgbs[:, s, 1]
            rgb[ray_idx, 2] = rgb[ray_idx, 2] + w * cur_rgbs[:, s, 2]
            depth[ray_idx] = depth[ray_idx] + (w * z).reshape(-1)[0]
            opacity[ray_idx] = opacity[ray_idx] + w.reshape(-1)[0]
            T = T * (1.0 - a[0, 0])

    return rgb, depth, opacity





