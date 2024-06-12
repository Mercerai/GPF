import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import get_embedder
# from models.point2sdf import PointsToSurfModel, PointsbasedSDFModel
import frnn
# from models.pointcloudmodels import Point_Agg, Point_Feature_Query
from utils.utils import weights_init
import open3d as o3d
import numpy as np
#

def temperature_scaled_softmax(logits, temperature=1.0, dim=-1):
    assert temperature > 0, "Temperature must be positive"
    
    # Scale the logits by the temperature
    scaled_logits = logits / temperature
    
    # Apply softmax
    return F.softmax(scaled_logits, dim=dim)

class NeuralPoints(nn.Module):
    def __init__(self, args, hid_n= 32):
        super(NeuralPoints, self).__init__()
        self.interp_method = args.model.interp_method
        self.points = None
        self.low_feats, self.high_feats, self.feat0, self.feat1, self.feat2 = None, None, None, None, None
        self.space_feature = None
        self.space_feature_dim = 0
        self.local_pts_num = args.model.K
        self.sh_degree = args.model.sh_degree
        self.feat_level = [0, 1, 2]

        self.low_agg = low_level_aggregator(feat_ch=8+3+1, hid_n=hid_n)
        self.high_agg = high_level_aggregator(feat_ch=32+3+1, hid_n=hid_n)

        # self.points_agg = Point_Agg(3, d_out=32, decimation=4, num_neighbors=16, device=args.device)
        # self.points_query = Point_Feature_Query(d_out=hid_n, feat_dim=hid_n * 2)
        positional_embedding=False
        if positional_embedding:
            self.embed_xyz_fn, embed_xyz_dim = get_embedder(args.model.multires_xyz, input_dim=3)
            self.embed_local_xyz_fn, embed_local_xyz_dim = get_embedder(args.model.multires_xyz, input_dim=3)
            self.embed_dist_fn, embed_dist_dim = get_embedder(args.model.multires_dist, input_dim=1)
            embed_pos_dim = embed_dist_dim + embed_xyz_dim + embed_local_xyz_dim
        else:
            embed_pos_dim = 1+3+3

        sgm_n = 32
        hid_n_in_network = sgm_n
        self.sigmanet = SigmaNet(pos_dim=embed_pos_dim, embed_feat_dim=hid_n, hid_n=hid_n_in_network, interp_method=args.model.interp_method)
        # self.sigmanet = SigmaNet_Only_pos(pos_dim=embed_pos_dim, embed_feat_dim=hid_n, hid_n=32,
        #                          interp_method=args.model.interp_method)
        # self.sigmanet = PointsToSurfModel(input_dims_per_point=3 + hid_n, net_size_max=256, num_points=self.local_pts_num)
        # self.sigmanet = PointsbasedSDFModel(input_dims_per_point=3 + hid_n, net_size_max=32, feat_output=sgm_n, num_points=self.local_pts_num)

        self.colornet = ColorNet(pos_dim=embed_pos_dim, embed_feat_dim=hid_n, hid_n=32,
                                 use_viewdirs=args.model.use_viewdirs, interp_method=args.model.interp_method)

        # self.colornet = ColorWeightsNet(pos_dim=embed_pos_dim, embed_feat_dim=hid_n, hid_n=hid_n_in_network, sgm_n=sgm_n,
        #                                 use_viewdirs=args.model.use_viewdirs, interp_method=args.model.interp_method)


    def forward(self, sampled_points_info):
        #
        indices = sampled_points_info['indices']  # [B, valid_pts, K]
        weights = sampled_points_info["weights"][..., None]
        xyz = sampled_points_info["world_xyz"] # [B, valid_pts, 3]
        rays_d = sampled_points_info["rays_d"] #  [B, valid_pts, K]

        extrinsics = sampled_points_info["extrinsics"].squeeze()
        cam_points = (torch.cat([self.points, torch.ones_like(self.points[..., 0:1])], dim=-1) @ (
            extrinsics.transpose(1, 0)))[..., :3]
        B, N, K = indices.shape
        selected_high_feats = torch.index_select(self.high_feats, dim=-2, index=indices.flatten()).view(B, N, K, -1)
        selected_low_feats = torch.index_select(self.low_feats, dim=-2, index=indices.flatten()).view(B, N, K, -1)  # B, 1RSRK, C
        selected_colors = torch.index_select(self.points_rgb, dim=-2, index=indices.flatten()).view(B, N, K, -1)
        neighbors = torch.index_select(cam_points, dim=-2, index=indices.flatten()).view(B, N, K, -1)
        xyz = (torch.cat([xyz, torch.ones_like(xyz[..., 0:1])], dim=-1) @ (extrinsics.transpose(1, 0)))[..., :3]
        xyz = xyz.reshape(B, N, 1, 3).repeat(1, 1, K, 1)

        # normalize_local_point = neighbors - xyz
        # radius = torch.linalg.norm(normalize_local_point, ord=2, dim=-1, keepdim=True)
        # max_radius = torch.max(radius, dim=-2, keepdim=True)[0]  # [B, R, SR, 1, 1]
        # normalize_local_point = normalize_local_point / max_radius
        # normalize_local_point = torch.cat([normalize_local_point, selected_high_feats], dim=-1)
        # sdf, sigmafeats = self.sigmanet(normalize_local_point)  # [B, R, SR, 1]
        # sdf = sdf * max_radius.squeeze(-1)

        dist = torch.norm((neighbors - xyz), p=2, dim=-1, keepdim=True)
        local_xyz = neighbors - xyz
        embed_pos = torch.cat([dist, local_xyz, neighbors], dim=-1)

        sdf, sigmafeats = self.sigmanet(embed_pos, selected_high_feats, weights=weights)

        # def forward(self, pos, img_feats, sigma_feats=None, view_dirs=None, weights=None):

        # rgb = self.colornet(embed_pos, selected_low_feats, selected_colors, sigma_feats=sigmafeats, # pcd model needs sigmafeats.view(B, R, SR, 1, -1).repeat(1,1,1,K,1)
        #                       view_dirs=rays_d[...,None,:].repeat(1,1,K,1), weights=weights)
        rgb = self.colornet(embed_pos, selected_low_feats, sigma_feats=sigmafeats, # pcd model needs sigmafeats.view(B, R, SR, 1, -1).repeat(1,1,1,K,1)
                            view_dirs=rays_d[..., None, :].repeat(1, 1, K, 1), weights=weights)

        return rgb, sdf

    def set_points(self, points):
        self.points = points

    def update_points(self):
        self.points += self.delta_p.detach()

    def set_points_rgb(self, points_rgb):
        self.points_rgb = points_rgb

    def set_img_feats(self, feature_list):
        assert len(feature_list) == 3
        for i in range(len(feature_list)):
            self.__setattr__("feat" + str(i), feature_list[i])

    def get_points(self):
        return self.points

    def gen_points_neural_features(self, scores):
        # feat012: B,N,3,C(C0,C1,C2)
        #scores : B, N ,3
        assert self.points is not None, "set point first of all"
        if (self.feat0 is None) or (self.feat1 is None) or (self.feat2 is None):
            raise TypeError("features are None")
        scores = scores[..., None]
        low_feats, color_w = self.low_agg(self.feat0, scores)
        high_feats = self.high_agg(self.feat2, scores)

        rgb = self.feat0[..., -3:]

        # _, ind = torch.max(scores, dim=-2)
        _, ind = torch.max(color_w, dim=-2)
        ind = ind.squeeze()

        # score 的量级
        # scores = F.softmax(scores, dim=-2)
        # scores = scores/torch.sum(scores, dim=-2, keepdim=True)
        points_rgb = torch.sum(rgb * color_w, dim=-2)
        # points_rgb = rgb[..., 0, :]

          # [B, N, hid_n]
        # point_feats = self.points_agg(self.points, self.points)  # [B, N, 32]

        self.low_feats = low_feats
        self.points_rgb = points_rgb
        self.high_feats = high_feats

        # print(low_feats.shape)
        # print(low_feats[:,0])
        # print(high_feats[:,0])
        # print("low feats ", low_feats.mean(), "  ", low_feats.min(), "  ", low_feats.max())
        # print("high feats ", low_feats.mean(), "  ", high_feats.min(), "  ", high_feats.max())
        #self.point_feats = point_feats

    def release_cache(self):
        if self.low_feats is not None:
            del self.low_feats
            self.low_feats = None
        if self.high_feats is not None:
            del self.high_feats
        if self.points_rgb is not None:
            del self.points_rgb
            self.points_rgb = None
        if (self.feat0 is not None):
            del self.feat0
            del self.feat1
            del self.feat2
            self.feat0, self.feat1, self.feat2 = None, None, None
        if self.points is not None:
            del self.points
            self.points = None

    def deploy(self):
        # if not (self.low_feats and self.high_feats and self.points and self.points_rgb):
        #     raise AttributeError("Lack of Attribute in neural points")
        # if (self.feat0 and self.feat1 and self.feat2):
        #     del self.feat0, self.feat1, self.feat2
        self.low_feats = nn.Parameter(self.low_feats, requires_grad=True)
        self.high_feats = nn.Parameter(self.high_feats, requires_grad=True)
        self.points_rgb = nn.Parameter(self.points_rgb, requires_grad=True)
        # self.delta_p = nn.Parameter(torch.zeros_like(self.points), requires_grad=True)
        self.points = nn.Parameter(self.points, requires_grad=True)
#
    def compute_sdf(self, extrinsics, world_query_points, indices, depth, rays_o, rays_d, weights):
        """
        :param extrinsics: 4,4  :param query_points: B, R, SR, 3  :param depth: B, 1, R  :return: B, R, SR, 1
        """
        # local_pts_num = self.local_pts_num
        # # local_pts_num = 8
        # world_xyz = self.points
        # net = self.sigmanet
        # B, R, SR , K = rays_d.shape
        #
        # cam_points = (torch.cat([world_xyz, torch.ones_like(world_xyz[..., 0:1])], dim=-1) @ (
        #     extrinsics.transpose(1, 0)))[..., :3]
        #
        #
        # depth_z_vals = depth.reshape(B, R, 1)
        # world_center = rays_o[:, :, 0:1] + rays_d[:, :, 0:1] * (depth_z_vals[..., None])
        #
        # dist, indices, _, _ = frnn.frnn_grid_points(world_center.view(B, -1, 3),
        #                                             world_xyz,
        #                                             K=local_pts_num,
        #                                             r=1000,
        #                                             return_nn=False,
        #                                             return_sorted=True)
        #
        # # origin = torch.tensor([0., 0, 0]).float().to(world_xyz.device).reshape(1, 1, 3)
        # # cam_points = torch.cat([cam_points, origin], dim=1)
        # # num_pts = cam_points.shape[-2]
        # # indices[indices == -1] = (num_pts - 1)
        #
        # cam_query_points = (torch.cat([world_query_points, \
        #                                torch.ones_like(world_query_points[..., 0:1])], dim=-1) @ (
        #                         extrinsics.transpose(1, 0)))[..., :3]
        # selected_high_feats = torch.index_select(self.high_feats, dim=-2, index=indices.flatten()).view(B, R, 1, local_pts_num, -1)  # B, 1RSRK, 1
        #
        # cam_local_pcd = torch.index_select(cam_points, dim=-2, index=indices.flatten()).reshape(B, R, 1, local_pts_num, -1)
        # cam_query_points = cam_query_points.unsqueeze(-2).repeat(1, 1, 1, local_pts_num, 1)  # [B, R, SR, local_pts_num, 3]
        #
        # normalize_local_point = cam_local_pcd - cam_query_points
        # dist = torch.linalg.norm(normalize_local_point, ord=2, dim=-1, keepdim=True)
        # max_radius = torch.max(dist, dim=-2, keepdim=True)[0]  # [B, R, SR, 1, 1]
        # normalize_local_point = normalize_local_point / max_radius
        # normalize_local_point = torch.cat([normalize_local_point, selected_high_feats.repeat(1,1,SR,1,1)], dim=-1)
        # # normalize_local_point = torch.cat([normalize_local_point, selected_high_feats], dim=-1)
        #
        # out = net(normalize_local_point)  # [B, R, SR, 1]
        # out = out * max_radius.squeeze(-1)

        B, R, SR, K = indices.shape  # here the SR is the number of pts_mid on each ray (SR - 1)
        selected_high_feats = torch.index_select(self.high_feats, dim=-2, index=indices.flatten()).view(B, R, SR, K, -1)
        # cam_points = (torch.cat([self.points, torch.ones_like(self.points[..., 0:1])], dim=-1) @ (
        #     extrinsics.transpose(1, 0)))[..., :3]

        # points = torch.cat([cam_points, torch.zeros_like(cam_points[:, 0:1, :]).to(cam_points.device)], dim=-2)

        # n_pts = points.shape[-2]
        # indices = indices.clone()
        # indices[indices == -1] = (n_pts - 1)

        neighbors = torch.index_select(self.points, dim=-2, index=indices.flatten()).view(B, R, SR, K, -1)
        # world_query_points = (torch.cat([world_query_points, torch.ones_like(world_query_points[..., 0:1])], dim=-1) @ (extrinsics.transpose(1, 0)))[..., :3]
        world_query_points = world_query_points.reshape(B, R, SR, 1, 3).repeat(1, 1, 1, K, 1)

        dist = torch.norm((neighbors - world_query_points), p=2, dim=-1, keepdim=True)
        local_xyz = neighbors - world_query_points
        embed_pos = torch.cat([dist, local_xyz, neighbors, world_query_points], dim=-1)

        out, sigmafeats = self.sigmanet(embed_pos, selected_high_feats, weights=weights)

        return out, sigmafeats

    def compute_rgb(self, extrinsics, xyz, indices, rays_d, sgm_feats, weights):
        B, R, SR, K = indices.shape #here the SR is the number of pts_mid on each ray (SR - 1)
        # cam_points = (torch.cat([self.points, torch.ones_like(self.points[..., 0:1])], dim=-1) @ (extrinsics.transpose(1, 0)))[..., :3]
        # low_feats = torch.cat([self.low_feats, torch.zeros_like(self.low_feats[:, 0:1, :]).to(self.low_feats.device)], dim=-2)
        # points_rgb = torch.cat([self.points_rgb, torch.zeros_like(self.points_rgb[:, 0:1, :]).to(self.points_rgb.device)], dim=-2)
        # points = torch.cat([cam_points, torch.zeros_like(cam_points[:, 0:1, :]).to(cam_points.device)], dim=-2)

        # n_pts = points.shape[-2]
        # indices = indices.clone()
        # indices[indices == -1] = (n_pts - 1)

        selected_low_feats = torch.index_select(self.low_feats, dim=-2, index=indices.flatten()).view(B, R, SR, K, -1)  # B, 1RSRK, C
        selected_colors = torch.index_select(self.points_rgb, dim=-2, index=indices.flatten()).view(B, R, SR, K, -1)
        neighbors = torch.index_select(self.points, dim=-2, index=indices.flatten()).view(B, R, SR, K, -1)

        # xyz = (torch.cat([xyz, torch.ones_like(xyz[..., 0:1])], dim=-1) @ (extrinsics.transpose(1, 0)))[..., :3]
        xyz = xyz.reshape(B, R, SR, 1, 3).repeat(1, 1, 1, K, 1)

        dist = torch.norm((neighbors - xyz), p=2, dim=-1, keepdim=True)
        local_xyz = neighbors - xyz

        embed_pos = torch.cat([dist, local_xyz, neighbors, xyz], dim=-1)

        color = self.colornet(embed_pos, selected_low_feats, selected_colors, sigma_feats=sgm_feats,
                              view_dirs=rays_d, weights=weights)

        return color


class low_level_aggregator(nn.Module):
    def __init__(self, feat_ch=8+3, hid_n=16):
        super(low_level_aggregator, self).__init__()

        self.score_embedder, score_embed_dim = get_embedder(4, 1)
        feat_ch = feat_ch - 1 + score_embed_dim

        self.color_weights = nn.Sequential(
            nn.Linear(feat_ch, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # nn.Softmax(dim=-2),
        )

        self.global_feat = nn.Sequential(
            nn.Linear(feat_ch, hid_n),
            nn.ReLU(),
        )

        self.feat_weights = nn.Sequential(
            nn.Linear(hid_n, 1),
            nn.ReLU(),
            nn.Softmax(dim=-2),
        )

    def forward(self, feats0, scores):
        scores_embedded = self.score_embedder(scores)
        feats = torch.cat([feats0, scores_embedded], dim=-1)

        color_w = self.color_weights(feats)  # [B, N, 3, 1]
        color_w = temperature_scaled_softmax(color_w, 0.9, -2)
        color_w = temperature_scaled_softmax(color_w * scores, 1, -2) # adjust the raw scores
        global_feats = self.global_feat(feats)
        feat_w = self.feat_weights(global_feats)
        global_feats = torch.sum((global_feats * feat_w), dim=-2)  # [B,N,16]
        # global_feats = torch.cat([global_feats, color], dim=-1)  # [B, N, 16+3]
        # return self.fc(global_feats)  # [B, N, 2*hid_n]
        return global_feats, color_w


class high_level_aggregator(nn.Module):
    def __init__(self, feat_ch=32+3+1, hid_n=32):
        super(high_level_aggregator, self).__init__()

        self.score_embedder, score_embed_dim = get_embedder(4, 1)
        feat_ch = feat_ch - 1 + score_embed_dim

        self.global_feat = nn.Sequential(
            nn.Linear(feat_ch * 3, 32),
            nn.ReLU(),
            nn.Linear(32, hid_n),
            nn.ReLU(),
        )

        self.global_w = nn.Sequential(
            nn.Linear(hid_n, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softmax(dim=-2),
        )

    def forward(self, feat2, scores):
        # x [B, N, 3, C+3]
        B, N, S, _ = feat2.shape
        score_embedded = self.score_embedder(scores)
        global_feats = torch.cat([feat2, score_embedded], dim=-1)

        var = torch.var(global_feats, dim=-2).reshape(B, N, 1, -1).repeat(1, 1, S, 1)
        avg = torch.mean(global_feats, dim=-2).reshape(B, N, 1, -1).repeat(1, 1, S, 1)

        feats = self.global_feat(torch.cat([var, avg, global_feats], dim=-1))
        w = self.global_w(feats)
        feats = torch.sum(feats * w, dim=-2)
        return feats  # [B, N, hid]

class SigmaNet(nn.Module):
    def __init__(self, pos_dim, embed_feat_dim, hid_n=32, interp_method="attention"):
        super(SigmaNet, self).__init__()
        self.local_positional_encoding = nn.Sequential(nn.Linear(pos_dim, hid_n), nn.ReLU())

        self.weights_from_only_pos = nn.Sequential(nn.Linear(hid_n, hid_n // 2), nn.ReLU(), nn.Linear(hid_n // 2, 1))
        self.position_weights = nn.Sequential(nn.Linear(hid_n + embed_feat_dim, hid_n), nn.ReLU(), nn.Linear(hid_n, 1))

        self.mlp1 = nn.Sequential(nn.Linear(embed_feat_dim + hid_n, hid_n), nn.ReLU(),
                                  nn.Linear(hid_n, hid_n), nn.ReLU())

        self.interp_method = interp_method
        if interp_method == "attention":
            self.weights = nn.Sequential(nn.Linear(hid_n, 1), nn.Softmax(dim=-2))

        self.final = nn.Sequential(
            nn.Linear(hid_n, hid_n),
            nn.ReLU(),
            nn.Linear(hid_n, 1),
            nn.Softplus()
        )

    def forward(self, embed_pos, high_feats, weights=None):
        pos_enc = self.local_positional_encoding(embed_pos)
        weights_only_pos = temperature_scaled_softmax(self.weights_from_only_pos(pos_enc), temperature=0.9, dim=-2)

        feat = torch.cat([high_feats, pos_enc], dim=-1)

        pos_weights = temperature_scaled_softmax(self.position_weights(feat), temperature=0.9, dim=-2)
        pos_weights = temperature_scaled_softmax(pos_weights * weights_only_pos, temperature=1, dim=-2)

        sgm_feat = self.mlp1(feat)

        if self.interp_method == "attention":
            # w = self.weights(sgm_feat)
            # agg_sgm_feat = torch.sum(sgm_feat * w, dim=-2)
            agg_sgm_feat = torch.sum(sgm_feat * pos_weights, dim=-2)
        elif self.interp_method == "inv_dis_weights":
            assert weights is not None
            agg_sgm_feat = torch.sum(weights * sgm_feat, dim=-2)
        else:
            raise NotImplementedError

        sigmas = self.final(agg_sgm_feat)

        return sigmas, sgm_feat

class SigmaNet_Only_pos(nn.Module):
    def __init__(self, pos_dim, embed_feat_dim, hid_n=32, interp_method="attention"):
        super(SigmaNet_Only_pos, self).__init__()
        self.local_positional_encoding = nn.Sequential(nn.Linear(pos_dim, hid_n), nn.ReLU())
        self.mlp1 = nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU())

        self.interp_method = interp_method
        if interp_method == "attention":
            self.weights = nn.Sequential(nn.Linear(hid_n, 1), nn.ReLU(), nn.Softmax(dim=-2))

        self.final = nn.Sequential(
            nn.Linear(hid_n, 1),
            nn.Softplus()
        )

    def forward(self, embed_pos, interp_method="attention", weights=None):

        pos_enc = self.local_positional_encoding(embed_pos)

        sgm_feat = self.mlp1(pos_enc)
        weights = self.weights(sgm_feat)
        agg_feat = torch.sum(sgm_feat * weights, dim=-2)

        sigmas = self.final(agg_feat)
        # sgm_feat = sgm_feat[..., None, :].repeat(1,1,1,8,1)
        # sigmas = torch.sum(sigmas * weights, dim=-2)

        return sigmas

class ColorNet(nn.Module):
    def __init__(self, pos_dim, embed_feat_dim, hid_n=32, sgm_n=32, use_viewdirs=True, interp_method="attention", sh_d=3):
        super(ColorNet, self).__init__()
        self.use_viewdirs = use_viewdirs
        self.interp_method = interp_method
        self.pos_enc = nn.Sequential(
            nn.Linear(pos_dim, hid_n),
            nn.ReLU(),
        )

        self.weights_from_only_pos = nn.Sequential(nn.Linear(hid_n, hid_n // 2), nn.ReLU(), nn.Linear(hid_n // 2, 1))
        self.position_weights = nn.Sequential(nn.Linear(hid_n + embed_feat_dim, hid_n), nn.ReLU(), nn.Linear(hid_n, 1))
        

        self.sh_d = sh_d
        if use_viewdirs:
            view_ch = 3
            ## remove view dir here, add it to the final mlp ##
            self.views_linears = nn.ModuleList([nn.Linear(embed_feat_dim +  sgm_n + hid_n, hid_n)])
            self.mlp1 = nn.Sequential(
                nn.Linear(hid_n, hid_n),
                nn.ReLU()
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(hid_n + embed_feat_dim + sgm_n + 3, hid_n),
                nn.ReLU()
            )

        if interp_method == "attention":
            self.weights = nn.Sequential(
                nn.Linear(hid_n, 1),
                nn.Softmax(dim=-2)
            )

        self.consider_view = nn.Sequential(
            nn.Linear(hid_n + view_ch, hid_n),
            nn.ReLU()
        )

        self.to_color = nn.Sequential(
            nn.Linear(hid_n, 3),
            nn.Sigmoid()
        )

    def forward(self, pos, img_feats, sigma_feats = None, view_dirs=None, weights=None):
        local_enc = self.pos_enc(pos)

        weights_only_pos = temperature_scaled_softmax(self.weights_from_only_pos(local_enc), temperature=0.9, dim=-2)

        feat = torch.cat([local_enc, img_feats], dim=-1)
        pos_weights = temperature_scaled_softmax(self.position_weights(feat), temperature=0.9, dim=-2)
        pos_weights = temperature_scaled_softmax(pos_weights * weights_only_pos, temperature=1, dim=-2)

        if self.use_viewdirs:
            h = torch.cat([img_feats, sigma_feats, local_enc], dim=-1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            color_feat = self.mlp1(h)

        else:
            color_feat = self.mlp1(torch.cat([img_feats, local_enc], dim=-1))

        if self.interp_method == "attention":
            # w = self.weights(color_feat)
            # agg_feat = torch.sum(w * color_feat, dim=-2)
            agg_feat = torch.sum(pos_weights * color_feat, dim=-2)
        # elif self.interp_method == "inv_dis_weights":
        #     assert weights is not None
        #     agg_feat = torch.sum(weights * color_feat, dim=-2)
        else:
            raise NotImplementedError
        agg_feat = self.consider_view(torch.cat([agg_feat, view_dirs[:, :, 0, :]], dim=-1))
        color = self.to_color(agg_feat)

        return color

class ColorWeightsNet(nn.Module):
    def __init__(self, pos_dim, embed_feat_dim, hid_n=32, sgm_n=32, use_viewdirs=True, interp_method=None):
        super(ColorWeightsNet, self).__init__()
        self.use_viewdirs = use_viewdirs
        self.pos_enc = nn.Sequential(
            nn.Linear(pos_dim, hid_n),
            nn.ReLU()
        )

        # self.pos_weights = nn.Sequential(
        #     nn.Linear(hid_n, 1),
        #     nn.ReLU()
        # )

        if self.use_viewdirs:
            view_ch = 3
            self.views_linears = nn.ModuleList([nn.Linear(hid_n + embed_feat_dim + 3 + view_ch + sgm_n, hid_n)])
            self.mlp1 = nn.Sequential(
                nn.Linear(hid_n, hid_n),
                nn.ReLU(),
                nn.Linear(hid_n, hid_n),
                nn.ReLU()
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(hid_n + embed_feat_dim + sgm_n + 3, hid_n),
                nn.ReLU(),
                nn.Linear(hid_n, hid_n),
                nn.ReLU()
            )
        self.weights = nn.Sequential(
            nn.Linear(hid_n, 1),
            nn.Softmax(dim=-2)
        )

    def forward(self, pos, img_feats,  neib_colors, view_dirs=None, weights=None, sigma_feats=None):
        local_enc = self.pos_enc(pos)

        if self.use_viewdirs:
            h = torch.cat([img_feats, local_enc, neib_colors, view_dirs, sigma_feats], dim=-1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            color_feat = self.mlp1(h)
        else:
            color_feat = self.mlp1(torch.cat([img_feats, local_enc, neib_colors], dim=-1))

        w = self.weights(color_feat)
        color = torch.sum(w * neib_colors, dim=-2)

        return color

