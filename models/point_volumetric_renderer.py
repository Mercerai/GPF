import torch
import frnn
import torch.nn as nn
import torch.nn.functional as F
from utils.rend_utils import visibility_score, batchify_neural_points_query, sdf_to_alpha, sdf_to_sigma
from torchvision.transforms import Resize
from models.volume_render_composite import composite
from models.model import FeatureNet
from models.neural_points3 import NeuralPoints
import open3d as o3d
class Point_Volumetric_Renderer(nn.Module):
    def __init__(self, args, K=8, radius=100, valid_src_imgs=3, ln_s = 0.1887, **dummy_dict):
        super(Point_Volumetric_Renderer, self).__init__()
        self.args = args
        self.feature_net = FeatureNet()
        self.neural_points = NeuralPoints(args=args) #aggregate selected features of points to get color for each sampling point
        self.points_feature = None # from get point cloud feature
        self.K = K
        self.radius = radius

        self.valid_src_imgs = valid_src_imgs
        self.feat_level = [0, 1, 2]
        self.feat_scales = [1., 0.5, 0.25]
        #
        self.speed_factor = 1.0
        self.ln_s = nn.Parameter(torch.Tensor([ln_s]), requires_grad=True)
        self.gamma = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

    def generate_neural_points_features(self, src_imgs, intrinsics, extrinsics, depths):
        assert self.get_points() is not None, "lack of points scaffold"

        scores, indices, vmask, _ = self.compute_visibility(intrinsics, extrinsics, depths)
        B, N, Sources = scores.shape
        ### filter out points out of boundarys
        mask = (scores[..., 0:1] > 1e-3)
        scores = scores[mask.repeat(1,1,Sources)].reshape(B, -1, Sources)
        indices = indices[mask.repeat(1,1,Sources)].reshape(B, -1, Sources)
        f_points = self.get_points()[mask.repeat(1,1,3)].reshape(1,-1,3)
        self.set_points(f_points)

        feat_dict = self.get_image_features(src_imgs)
        img_feats = []
        H, W = src_imgs.shape[-2], src_imgs.shape[-1]

        resize = Resize(size=(H // 4, W // 4))
        temp = []
        for j in range(src_imgs.size(1)):
            temp.append(resize(src_imgs[:, j]))
        resized_src_imgs = torch.stack(temp, dim=1)
        for i in self.feat_level:
            key = "feat" + str(i)
            feat = feat_dict[key]
            if i == 0:
                feat = torch.cat([feat, src_imgs], dim=2)  # [B,S,8+3,H,W]
            # elif i == 1:
            #     feat = torch.cat([feat, resized_src_imgs], dim=2)  # [B,S,16+3,H,W]
            elif i == 2:
                feat = torch.cat([feat, resized_src_imgs], dim=2)  # [B,S,32+3,H,W]
            extracted_feat = self.extract_2d(feat, intrinsics, extrinsics, self.feat_scales[i])
            img_feats.append(extracted_feat)

        s_img_feats = []
        for feat in img_feats:
            s_img_feats.append(self.gather_features(feat, indices))  # B,N,3,C

        self.neural_points.set_img_feats(s_img_feats)
        self.neural_points.gen_points_neural_features(scores)

    def neural_points_forward(self, sampled_points_info):
        sampled_points = sampled_points_info["world_xyz"]

        B, N, _ = sampled_points.shape

        indices, weights, distances = self.neighbor_search(sampled_points)

        sampled_points_info["indices"] = indices.view(B, N, -1)
        sampled_points_info["weights"] = weights.view(B, N, -1)
        sampled_points_info["distances"] = distances.view(B, N, -1)

        raw = batchify_neural_points_query(self.neural_points.forward, sampled_points_info, chunk=self.args.training.rays_chunk*64)
        return raw
    
    def neural_points_forward_parallel(self, sampled_points_info):
        sampled_points = sampled_points_info["world_xyz"]
        B, R, SR, _ = sampled_points.shape
        device = sampled_points.device
        sampled_points = sampled_points.view(1, -1, 3)
        indices, weights, distance = self.neighbor_search(sampled_points)
        indices = indices.to(device)
        valid_points = (indices != -1).float()
        valid_points = torch.prod(valid_points, dim=-1, keepdim=False).to(dtype=torch.bool) #[B, R*SR]

        # print("valid points ", valid_points)
        ray_indices = torch.arange(0, R).reshape(1, R, 1).repeat(B, 1, SR).reshape(B, R*SR).to(device)
        # print(ray_indices)
        ray_indices = ray_indices[:, valid_points.flatten()]
        # print(" =  = = = == ", ray_indices.shape)
        alive_ray_index = torch.unique(ray_indices.reshape(-1))
        alive_points = sampled_points[:, valid_points.flatten()] #[B, valid_pts, 3]
        alive_indices = indices[:, valid_points.flatten()] #[B, valid_pts, K]
        # print(f"alive indice {alive_indices.shape}")
        alive_weights = weights[:, valid_points.flatten()] #[B, valid_pts, K]
        sampled_points_info["rays_d"] =  sampled_points_info["rays_d"].reshape(B, -1, 3)[:, valid_points.flatten()]
        sampled_points_info["dist"] = sampled_points_info["dist"].reshape(B, -1)[:, valid_points.flatten()]
        # sampled_points_info["rays_o"] = torch.index_select(sampled_points_info["rays_o"].reshape(B, -1, 3), index=valid_points.flatten(), dim=-2) #[B, valid_pts, 3]
        sampled_points_info["valid_points"] = valid_points
        sampled_points_info["ray_indices"] = ray_indices
        sampled_points_info["alive_ray_index"] = alive_ray_index
        sampled_points_info["indices"] = alive_indices
        sampled_points_info["weights"] = alive_weights
        sampled_points_info["alive_points"] = alive_points
        sampled_points_info["z_vals"] = sampled_points_info["z_vals"].reshape(B, -1)[..., valid_points.flatten()]


        raw = batchify_neural_points_query(self.neural_points.forward, sampled_points_info, chunk=self.args.training.rays_chunk*128)
        return raw, sampled_points_info

    #
    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward(self, src_imgs, sampled_points_info, intrinsics, extrinsics, depths):
        self.generate_neural_points_features(src_imgs, intrinsics, extrinsics, depths)
        rgbs, sigmas = self.neural_points_forward(sampled_points_info)
        R, SR = sampled_points_info["R"], sampled_points_info["SR"]
        # mask = (sampled_points_info["distances"] > 10).float()
        # mask = torch.prod(mask, dim=-1, keepdim=False).flatten().to(dtype=torch.bool)
        # rgbs[:, mask] = 0.
        # sigmas[:, mask] = 0.
        rgbs, sigmas = rgbs.reshape(-1, R, SR, 3), sigmas.reshape(-1, R, SR, 1)
        ret = self.raw2output(rgbs, sigmas.squeeze(-1), sampled_points_info["dist"].reshape(-1, R, SR), sampled_points_info["z_vals"].reshape(-1, R, SR), self.args.training.whitebg)
        # ret = self.composite_rays(rgbs, sigmas, sampled_points_info)
        return ret

    def composite_rays(self, rgbs, sigmas, sampled_points_info):
        B, R, SR, _ = sampled_points_info["world_xyz"].shape
        device = sampled_points_info["rays_d"].device
        opacity = torch.zeros(R, device=device)
        rgb = torch.zeros(R, 3, device=device)
        depth = torch.zeros(R, device=device)

        steps = []
        start_indices = []
        alive_ray_index = sampled_points_info["alive_ray_index"]
        ray_indices = sampled_points_info["ray_indices"].squeeze()
        for index in alive_ray_index:
            cur_indices = torch.nonzero(ray_indices == index)
            steps.append(len(cur_indices))
            start_indices.append(cur_indices[0])
        steps = torch.LongTensor(steps).to(device)
        start_indices = torch.LongTensor(start_indices).to(device)

        composite(sigmas.squeeze(), rgbs.squeeze(), start_indices, steps, sampled_points_info["dist"].squeeze(), sampled_points_info["z_vals"].squeeze(), sampled_points_info["alive_ray_index"].squeeze(), opacity, depth, rgb)

        return {'rgb': rgb, 'depth': depth, 'acc_map': opacity, "alpha":None, "weights":None, "depth weights":None}



    def get_points(self):
        return self.neural_points.get_points()

    def get_points_rgb(self):
        return self.neural_points.points_rgb

    def release_neural_pts_cache(self):
        self.neural_points.release_cache()

    def neighbor_search(self, query_points):
        assert len(query_points.shape) == 3, "dimension of query points is not 3"
        dist, indices, _, _ = frnn.frnn_grid_points(query_points,
                              self.neural_points.get_points(),
                              K=self.K,
                              r=self.radius,
                              return_nn=False,
                              return_sorted=True) #(1, P1, K)

        dist = dist.detach()
        indices = indices.detach()
        dist = dist.sqrt()
        weights = 1 / (dist + 1e-7)
        weights = weights / torch.sum(weights, -1, keepdim=True) #(1, P1, K)

        return indices, weights, dist

    def get_point_cloud_feature(self):
        pass

    def get_image_features(self, images):
        B, S, C, H, W = images.shape
        images = images.reshape(-1, C, H, W)
        feat2, feat1, feat0 = self.feature_net(images)

        feat0 = feat0.reshape(B, S, -1, H, W)
        feat1 = feat1.reshape(B, S, -1, H//2, W//2)
        feat2 = feat2.reshape(B, S, -1, H//4, W//4)

        feats = {"feat0":feat0, "feat1":feat1, "feat2":feat2}
        return feats

    def set_points(self, points):
        assert len(points.shape) == 3, "in set points, dimension of points is not 3"
        self.neural_points.set_points(points)

    def set_neural_points_rgb(self, rgb):
        self.neural_points.set_points_rgb(rgb)

    def set_K_and_r(self, k, r):
        self.radius = r
        self.K = k # neighbor search

    def set_valid_src_imgs(self, n):
        self.valid_src_imgs = n

    def compute_visibility(self, intrinsics, extrinsics, depths):
        #int, ext, depths of src imgs
        points = self.neural_points.get_points()
        assert points is not None, "lack of points before computing visibility"

        n_srcs = intrinsics.shape[1]
        scores = []
        for i in range(n_srcs):
            int, ext, depth = intrinsics[:, i], extrinsics[:, i], depths[:, i]
            scores.append(visibility_score(points, int, ext, depth))
        scores = torch.cat(scores, dim=-1) # [B, N, S]
        values, topk_indices = torch.topk(scores, dim=-1, k=self.valid_src_imgs, largest=True)

        neg_filter = (scores < values[..., -1, None]) # smaller than the third rank of value
        mask = ~neg_filter

        return values, topk_indices.long(), mask, scores

    def extract_2d(self, img_feats, intrinsics, extrinsics, feat_scale=1):
        points = self.neural_points.get_points()
        assert points is not None, "lack of points before extract 2d features"

        points = torch.cat([points, torch.ones_like(points[..., 0:1])], dim=-1)
        B, S, _, H, W = img_feats.shape
        # H, W = H*feat_scale, W*feat_scale
        ret_feat = []
        for i in range(S):
            feat = img_feats[:, i]
            int, ext = intrinsics[:, i], extrinsics[:, i]
            points_cam = (points @ ext.transpose(-1, -2))[..., :3]
            int_ = int.clone()
            int_[..., :2] = int_[..., :2] * feat_scale
            points_pixel = points_cam @ int_.transpose(-1, -2)
            grid = points_pixel[..., 0:2] / torch.clamp_min(points_pixel[..., 2:3], 1e-6)
            grid[..., 0], grid[..., 1] = (grid[..., 0]) / (W - 1), (grid[..., 1]) / (H - 1)
            grid = 2.*grid - 1. #[B, N, 2]

            sampled_feat = F.grid_sample(feat, grid[:, None], align_corners=True, padding_mode="border",
                                         mode="bilinear").permute(0, 2, 3, 1)[:, 0]

            ret_feat.append(sampled_feat)

        return torch.stack(ret_feat, dim=-2) #[B, N, S, C]

    def gather_features(self, points_features, visibility_indices):
        # points features [B, N, S, F]
        # visibility indices [B, N, 3]
        # select features according to sorted visibility
        B,N,S,F = points_features.shape
        indices = visibility_indices.view(B, N, -1, 1).repeat(1,1,1,F)
        return torch.gather(points_features, dim=2, index=indices) #B, N, 3, F


    def raw2output(self, rgb, sigma, dist, z_vals=None, white_bkgd=False):
        """Transforms model's predictions to semantically meaningful values.
         Param:
             raw: [num_rays, num_samples along ray, 4]. Prediction from model.
             z_vals: [num_rays, num_samples along ray]. Integration time.
             rays_d: [num_rays, 3]. Direction of each ray.
         Returns:
             rgb_map: [(B), num_rays, 3]. Estimated RGB color of a ray.
             disp_map: [(B), num_rays]. Disparity map. Inverse of depth map.
             acc_map: [(B), num_rays]. Sum of weights along each ray.
             weights: [(B), num_rays, num_samples]. Weights assigned to each sampled color.
             depth_map: [(B), num_rays]. Estimated distance to object.
         """
        raw2alpha = lambda raw: 1. - torch.exp(-raw * dist)
        # raw2alpha = lambda raw: 1. - torch.exp(-raw)
        alpha = raw2alpha(sigma)  # [B, N_rays, N_samples]

        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]
        T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
        weights = alpha * T
        # print(weights.shape, " ============= ", type(weights))
        # print(rgb.shape, " ================ ", type(rgb))
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [B, N_rays, 3]
        # rgb_map = torch.mean(rgb, dim=-2)
        acc_map = torch.sum(weights, dim=-1) #[B, N_rays]
        if z_vals is not None:
            # print("z vals :", z_vals)
            # depth_weights = F.softmax(weights, dim=-1)
            depth_weights = weights / (weights.sum(-1, keepdim=True) + 1e-10)
            depth_map = torch.sum(depth_weights * z_vals.detach(), -1)
            # depth_map = depth_map/depth_map.max()

        else:
            depth_map = None

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])
            depth_map = depth_map + (1. - acc_map)
        else:
            # depth_map = depth_map - (1. - acc_map) * depth_map
            depth_map = depth_map

        return {'rgb': rgb_map, 'depth': depth_map, 'acc_map': acc_map, "alpha":alpha, "weights":weights, "depth weights":depth_weights}


class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        print(" asdf ",0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))
        print(" asdf asdf ", -sdf.abs() / beta)
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta