import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Union, Dict

def print_args(args: Any) -> None:
    """Utilities to print arguments

    Args:
        args: arguments to print out
    """
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")

def compute_loss(ret, rgb_gt, depth_gt, mask_gt, z_vals, global_step):
    """
    :param ret, :param rgb_gt:  [B, N, 3]
    :return:
    """
    rgb_loss = torch.tensor(0)
    depth_loss = torch.tensor(0)
    weights_loss = torch.tensor(0)
    sparse_loss = torch.tensor(0)
    mask_loss = torch.tensor(0)
    # rgb, depth, mask, sdf, alpha, weights, depth_weights = ret["rgb"], ret["depth"], ret["acc_map"], ret["sdf"], ret["alpha"], ret["weights"], ret["depth weights"]
    rgb, depth, mask, weights, alpha, depth_weights = ret["rgb"], ret["depth"], ret["acc_map"], ret["weights"], ret["alpha"], ret["depth weights"]
    mask_gt = mask_gt.reshape(-1, 1)
    mask_sum = mask_gt.sum()
    # rgb_error = (rgb.reshape(-1,3) - rgb_gt.reshape(-1, 3)) * mask_gt
    rgb_error = (rgb.reshape(-1,3) - rgb_gt.reshape(-1, 3))
    # rgb_loss = torch.sum(rgb_error ** 2) / mask_sum
    # rgb_loss = torch.sum(rgb_error ** 2) / int(mask_gt.shape[0])
    rgb_loss = F.l1_loss(rgb_error, torch.zeros_like(rgb_error).to(rgb.device), reduction="sum") / int(mask_gt.shape[0])
    # rgb_loss = F.l1_loss(rgb_error, torch.zeros_like(rgb_error).to(rgb.device), reduction="sum") / mask_sum

    depth_error = (depth.reshape(-1, 1) - depth_gt.reshape(-1, 1)) * mask_gt
    B,R,SR = alpha.shape
    # d = depth_gt.reshape(-1, 1).repeat(1, SR).reshape(1,R,SR)
    # usdfgt = (d - z_vals)
    # sdf_error = (sdf - usdfgt) * (mask_gt.reshape(1, R, 1).repeat(1, 1, SR))
    # # weights_loss = torch.sum(alpha_error ** 2)/mask_sum
    # weights_loss = F.l1_loss(sdf_error, torch.zeros_like(sdf_error).to(rgb.device), reduction="sum") / mask_sum

    if global_step % 500 == 0:
        # print(" usdfgt ", usdfgt.view(-1)[:SR])
        # print(" sdf ", sdf.view(-1)[:SR])
        print(" alpha ", alpha.view(-1)[:SR])
        print(" weights ", weights.view(-1)[:SR])
        print(" depth weights ", depth_weights.view(-1)[:SR])
        print(" depth ", depth.reshape(-1, 1)[mask_gt][:10])
        print(" depth gt ", depth_gt.reshape(-1, 1)[mask_gt][:10])
        print(" depth error ", depth_error[mask_gt][:10])

    depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error).to(depth.device), reduction="sum") / mask_sum
    # depth_error = depth.reshape(-1, 1) - depth_gt.reshape(-1, 1)
    # depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error).to(depth.device), reduction="mean")
    mask_loss = F.binary_cross_entropy(mask.reshape(-1, 1).clip(1e-7, 1.0 - 1e-7), mask_gt.float())
    # sparse_loss = torch.mean(torch.log(mask.view(-1).clip(1e-3, 1.0 - 1e-3)) + torch.log(1 - mask.view(-1).clip(1e-3, 1.0 - 1e-3)))
    sparse_loss = torch.mean((-mask.reshape(-1, 1).clip(1e-7, 1.0 - 1e-7) * torch.log(mask.reshape(-1, 1).clip(1e-7, 1.0 - 1e-7))))
    loss = {"rgb loss":rgb_loss, "depth loss":depth_loss, "mask loss":mask_loss, "weights loss":weights_loss, "sparse loss":sparse_loss}
    # print(f"rgb loss {rgb_loss}, depth loss{depth_loss}, mask loss {mask_loss}")
    return loss

def compute_loss2(ret, rgb_gt, depth_gt, mask_gt, z_vals, global_step):
    """
    :param ret, :param rgb_gt:  [B, N, 3]
    :return:
    """
    rgb_loss = torch.tensor(0)
    depth_loss = torch.tensor(0)
    weights_loss = torch.tensor(0)
    sparse_loss = torch.tensor(0)
    mask_loss = torch.tensor(0)
    # rgb, depth, mask, sdf, alpha, weights, depth_weights = ret["rgb"], ret["depth"], ret["acc_map"], ret["sdf"], ret["alpha"], ret["weights"], ret["depth weights"]
    rgb, depth, mask, weights, alpha, depth_weights = ret["rgb"], ret["depth"], ret["acc_map"], ret["weights"], ret["alpha"], ret["depth weights"]
    mask_gt = mask_gt.reshape(-1, 1)
    mask_sum = mask_gt.sum()
    # rgb_error = (rgb.reshape(-1,3) - rgb_gt.reshape(-1, 3)) * mask_gt
    rgb_error = (rgb.reshape(-1,3) - rgb_gt.reshape(-1, 3))
    # rgb_loss = torch.sum(rgb_error ** 2) / mask_sum
    # rgb_loss = torch.sum(rgb_error ** 2) / int(mask_gt.shape[0])
    rgb_loss = F.l1_loss(rgb_error, torch.zeros_like(rgb_error).to(rgb.device), reduction="sum") / int(mask_gt.shape[0])
    # rgb_loss = F.l1_loss(rgb_error, torch.zeros_like(rgb_error).to(rgb.device), reduction="sum") / mask_sum
    depth_error = (depth.reshape(-1, 1) - depth_gt.reshape(-1, 1)) * mask_gt
    _,N = z_vals.shape

    depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error).to(depth.device), reduction="sum") / mask_sum

    # depth_error = depth.reshape(-1, 1) - depth_gt.reshape(-1, 1)
    # depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error).to(depth.device), reduction="mean")
    mask_loss = F.binary_cross_entropy(mask.reshape(-1, 1).clip(1e-7, 1.0 - 1e-7), mask_gt.float())
    # sparse_loss = torch.mean(torch.log(mask.view(-1).clip(1e-3, 1.0 - 1e-3)) + torch.log(1 - mask.view(-1).clip(1e-3, 1.0 - 1e-3)))
    # sparse_loss = torch.mean((-mask.reshape(-1, 1).clip(1e-7, 1.0 - 1e-7) * torch.log(mask.reshape(-1, 1).clip(1e-7, 1.0 - 1e-7))))
    loss = {"rgb loss":rgb_loss, "depth loss":depth_loss, "mask loss":mask_loss, "weights loss":weights_loss, "sparse loss":sparse_loss}
    # print(f"rgb loss {rgb_loss}, depth loss{depth_loss}, mask loss {mask_loss}")
    return loss

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

def batch_quat_to_rotmat(q, out=None):
    """
    quaternion a + bi + cj + dk should be given in the form [a,b,c,d]
    :param q:
    :param out:
    :return:
    """

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2 / torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out


def adjust_learning_rate(optimizer, learning_rate, current_step, max_step=200000, warm_up_end = 5000, learning_rate_alpha=0.05):
    if current_step < warm_up_end:
        learning_factor = current_step / warm_up_end
    else:
        alpha = learning_rate_alpha
        progress = (current_step - warm_up_end) / (max_step - warm_up_end)
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        # learning_factor = 1

    for g in optimizer.param_groups:
        g['lr'] = learning_rate * learning_factor
