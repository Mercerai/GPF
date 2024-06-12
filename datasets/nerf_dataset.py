import numpy as np
import os
import random
import json
import imageio
import torch
from .data_io import read_cam_file, read_image, read_map, read_pair_file, read_ply
from torch.utils.data import Dataset
from typing import List, Tuple

class NeRFDataset(Dataset):
    def __init__(self,
            data_path: str,
            point_path,
            num_views: int = 10,
            max_dim: int = -1,
            scene_list: str = '',
            cam_file: str = "transforms_test.json",
            image_folder: str = "images",
            depth_folder: str = "depth_gt",
            mask_folder: str = "masks",
            image_extension: str = ".png",
            robust_train: bool = False,
            pcd_downsample = 0,
            downsample_method = "uniform"
    ) -> None:
        super(NeRFDataset, self).__init__()
        self.data_path = data_path
        self.point_path = point_path
        self.num_views = num_views
        self.max_dim = max_dim
        self.robust_train = robust_train
        self.cam_file = cam_file

        self.depth_folder = depth_folder
        self.mask_folder = mask_folder
        self.image_folder = image_folder
        self.image_extension = image_extension
        self.pcd_downsample = pcd_downsample
        self.downsample_method = downsample_method

        self.metas: List[Tuple[str,int]] = []
        self.data_length = 200

        if os.path.isfile(scene_list):
            with open(scene_list) as f:
                scenes = [line.rstrip() for line in f.readlines()]
        for scene in scenes:
            for i in range(self.data_length):
                img_idx = i
                self.metas += [(scene, img_idx)]

        self.intrinsics_dict = {}
        self.extrinsics_dict = {}
        self.poses_dict = {}

        for scene in scenes:
            with open(os.path.join(self.data_path, scene, self.cam_file)) as fp:
                meta = json.load(fp)
            img, H, W = read_image(os.path.join(self.data_path, scene, self.image_folder, "r_0{}".format(image_extension)), -1)
            camera_angle_x = float(meta["camera_angle_x"])
            focal = .5 * W / np.tan(.5 * camera_angle_x)

            K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ], dtype=np.float32)

            self.tf = np. array(
                [[1., 0, 0,0],
                [0, -1., 0,0],
                [0, 0, -1,0],
                 [0,0,0,1]]
            , dtype=np.float32)

            self.intrinsics_dict[scene] = K
            poses = []
            extrinsics = []
            for frame in meta["frames"]:
                c2w = np.array(frame["transform_matrix"]).astype(np.float32)
                c2w = c2w @ self.tf
                poses.append(c2w)
                extrinsics.append(np.linalg.inv(c2w))
            self.extrinsics_dict[scene] = extrinsics
            self.poses_dict[scene] = poses

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scene, ref_idx = self.metas[idx]

        num_src_views = self.num_views
        cam_points = np.array([c2w[:3,3] for c2w in self.poses_dict[scene]])
        cam_point = self.poses_dict[scene][ref_idx][:3,3]
        distance = np.linalg.norm(cam_points - cam_point[np.newaxis], axis=-1)
        argsorts = distance.argsort()
        argsorts = argsorts[1:]
        src_imgs_idx = argsorts[:num_src_views].tolist()
        view_ids = [ref_idx] + src_imgs_idx

        if (self.point_path is not None) and (self.point_path != "None"):
            points_filename = os.path.join(self.point_path , "{}.ply".format(scene))
        else:
            points_filename = os.path.join(self.data_path, scene, "points.ply")
        points = read_ply(points_filename, self.pcd_downsample, self.downsample_method)

        images = []
        intrinsics = []
        extrinsics = []
        depth_min: float = -1.0
        depth_max: float = -1.0
        depths = []

        for view_index, view_id in enumerate(view_ids):
            img_filename = os.path.join(
                self.data_path, scene, self.image_folder,"r_{}{}".format(view_id, self.image_extension))

            mask_filename = os.path.join(
                self.data_path, scene, self.mask_folder, "r_{}{}".format(view_id, self.image_extension))

            # the_mask, _, _ = read_image(mask_filename)
            # the_mask = the_mask[:, :, np.newaxis]

            image, original_h, original_w = read_image(img_filename, self.max_dim)
            # image = image * the_mask
            images.append(image.transpose([2, 0, 1]))

            intrinsic = self.intrinsics_dict[scene].copy()
            extrinsic = self.extrinsics_dict[scene][view_id]

            intrinsic[0] *= image.shape[1] / original_w
            intrinsic[1] *= image.shape[0] / original_h
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            depth_gt_filename = os.path.join(self.data_path, scene, self.depth_folder, "r_{}.pfm".format(view_id))
            dpt = read_map(depth_gt_filename, self.max_dim).transpose([2, 0, 1]).copy()
            depths.append(dpt)

            if view_index == 0:  # reference view
                # Using `copy()` here to avoid the negative stride resulting from the transpose
                # Create mask from GT depth map
                ref_mask = (dpt >= depth_min)

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)
        depth_range = np.array([2., 6.], dtype=np.float32)

        return {
            # "view_ids": view_ids,
            "images": images,  # List[Tensor]: [N][3,Hi,Wi], N is number of images where N is num of views
            "points": points,  # Tensor: [N, 3]
            "intrinsics": intrinsics,  # Tensor: [N,3,3]
            "extrinsics": extrinsics,  # Tensor: [N,4,4]
            "depth_range": depth_range,  # Tensor [2]
            "depths_gt": depths,  #
            "ref_mask": ref_mask,  # Tensor: [1,H0,W0] if exists
            # "filename": os.path.join(scan, "{}", "{:0>8}".format(view_ids[0]) + "{}")
            "filename": os.path.join(scene, "r_{}.png".format(view_ids[0]))
        }

    def get_single_(self, scene, ref_view):
        """
        return Tensor of one single training sample
        """
        scene, ref_idx = scene, ref_view
        num_src_views = self.num_views
        cam_points = np.array([c2w[:3, 3] for c2w in self.poses_dict[scene]])
        cam_point = self.poses_dict[scene][ref_idx][:3, 3]
        distance = np.linalg.norm(cam_points - cam_point[np.newaxis], axis=-1)
        argsorts = distance.argsort()
        argsorts = argsorts[1:]
        src_imgs_idx = argsorts[:num_src_views].tolist()
        view_ids = [ref_idx] + src_imgs_idx

        if (self.point_path is not None) and (self.point_path != "None"):
            points_filename = os.path.join(self.point_path , "{}.ply".format(scene))
        else:
            points_filename = os.path.join(self.data_path, scene, "points.ply")
        points = read_ply(points_filename, self.pcd_downsample, self.downsample_method)
        points = torch.from_numpy(points).unsqueeze(0)

        images = []
        intrinsics = []
        extrinsics = []
        depth_min: float = 2.
        depth_max: float = 6.
        depths = []

        for view_index, view_id in enumerate(view_ids):
            img_filename = os.path.join(
                self.data_path, scene, self.image_folder,"r_{}{}".format(view_id, self.image_extension))

            mask_filename = os.path.join(
                self.data_path, scene, self.mask_folder, "r_{}{}".format(view_id, self.image_extension))

            # the_mask, _, _ = read_image(mask_filename)
            # the_mask = the_mask[:, :, np.newaxis]

            image, original_h, original_w = read_image(img_filename, self.max_dim)
            # image = image * the_mask
            image_to_add = image.transpose([2, 0, 1])
            image_to_add = torch.from_numpy(image_to_add)[None]
            images.append(image_to_add)

            intrinsic = self.intrinsics_dict[scene].copy()
            extrinsic = self.extrinsics_dict[scene][view_id]

            intrinsic[0] *= image.shape[1] / original_w
            intrinsic[1] *= image.shape[0] / original_h
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            depth_gt_filename = os.path.join(self.data_path, scene, self.depth_folder, "r_{}.pfm".format(view_id))
            dpt = read_map(depth_gt_filename, self.max_dim).transpose([2, 0, 1]).copy()
            dpt = torch.from_numpy(dpt)[None]
            depths.append(dpt)

            if view_index == 0:  # reference view
                # Using `copy()` here to avoid the negative stride resulting from the transpose
                # Create mask from GT depth map
                ref_mask = (dpt >= depth_min)

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)
        intrinsics, extrinsics = torch.from_numpy(intrinsics)[None], torch.from_numpy(extrinsics)[None]
        depth_range = np.array([2., 6.], dtype=np.float32)
        depth_range = torch.from_numpy(depth_range)[None]

        return {
            # "view_ids": view_ids,
            "images": images,  # List[Tensor]: [N][3,Hi,Wi], N is number of images where N is num of views
            "points": points,  # Tensor: [N, 3]
            "intrinsics": intrinsics,  # Tensor: [N,3,3]
            "extrinsics": extrinsics,  # Tensor: [N,4,4]
            "depth_range": depth_range,  # Tensor [2]
            "depths_gt": depths,  # Tensor: [1,H0,W0] if exists
            "ref_mask": ref_mask,  # Tensor: [1,H0,W0] if exists
            # "filename": os.path.join(scan, "{}", "{:0>8}".format(view_ids[0]) + "{}")
            "filename": [os.path.join(scene, "r_{}.png".format(view_ids[0]))]
        }


# if __name__ == "__main__":

