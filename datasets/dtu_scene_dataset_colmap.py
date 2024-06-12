import os
import numpy as np
import cv2 as cv
import torch
from tqdm import tqdm
from datasets.data_io import glob_imgs, read_image, glob_txts, read_cam_file, read_map, read_ply, glob_pfms

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class SceneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 scene_path,
                 points_path,
                 pcd_downsample: 1,
                 max_dim= -1,
                 split="entire",
                 light_idx = "",
                 cam_folder="cams",
                 depth_folder="depth_gt",
                 image_folder="image",
                 mask_folder="mask",
                 pair_path="pair.txt",
                 train_cameras=False,
                 downsample_method="voxel",
                 verbose=False):

        self.scene_path = scene_path
        self.points_path = points_path
        self.train_cameras = train_cameras

        self.points = read_ply(points_path, pcd_downsample, downsample_method=downsample_method)
        self.points = torch.from_numpy(self.points).float()

        image_dir = os.path.join(scene_path, image_folder)
        image_paths = sorted(glob_imgs(image_dir))
        mask_dir = os.path.join(scene_path, mask_folder)
        mask_paths = sorted(glob_imgs(mask_dir))
        depth_dir = os.path.join(scene_path, depth_folder)
        depth_paths = sorted(glob_pfms(depth_dir))
        # if len(depth_paths) == 0:

        n_images = len(image_paths)

        if split == "entire":
            self.mask = np.ones(len(image_paths)) == 1
        elif split == "train" or split == "val":
            val_img_path = os.path.join(self.scene_path, "val.txt")
            assert os.path.exists(val_img_path), "validation imgs is empty"
            with open(val_img_path, "r") as f:
                ss = f.read().rstrip()
                val_numbers = list(map(int, ss.split("\n")))
            if split == "train":
                self.mask = np.ones(len(image_paths)) == 1
                self.mask[val_numbers] = False
            else:
                self.mask = np.ones(len(image_paths)) == 0
                self.mask[val_numbers] = True

            if verbose == True and os.path.exists(
                os.path.join(self.instance_dir, "val_imgs.txt")
            ):
                with open(os.path.join(self.instance_dir, "val_imgs.txt"), "r") as f:
                    ss = f.read().rstrip()
                    val_imgs = ss.split("\n")
                for i, val_img_name in zip(val_numbers, val_imgs):
                    assert val_img_name == os.path.basename(image_paths[i])
        else:
            raise NotImplementedError

        n_images = len(image_paths)

        img, original_h, original_w = read_image(image_paths[0], max_dim)
        self.H, self.W, _ = img.shape
        print(f"In DTU Scene, H and W are {self.H}, {self.W}")

        cam_dir = os.path.join(scene_path, "cameras_sphere.npz")
        camera_dict = np.load(cam_dir)
        self.camera_dict = camera_dict
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        self.intrinsics_all = []
        self.c2w_all = []
        self.extrinsics_all = []
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics[0] *= self.W / original_w
            intrinsics[1] *= self.H / original_h
            self.intrinsics_all.append(intrinsics[:3, :3]) #3, 3
            self.c2w_all.append(pose) #4, 4
            self.extrinsics_all.append(np.linalg.inv(pose))


        self.intrinsics_all = [torch.from_numpy(self.intrinsics_all[i]).float()
                                for i in range(n_images)
                               if self.mask[i] == True
                               ]

        self.c2w_all = [torch.from_numpy(self.c2w_all[i]).float()
                        for i in range(n_images)
                        if self.mask[i] == True
                        ]

        self.extrinsics_all = [torch.from_numpy(self.extrinsics_all[i]).float()
                        for i in range(n_images)
                        if self.mask[i] == True
                        ]

        self.rgb_images = []
        for i, path in tqdm(enumerate(image_paths), desc="loading images..."):
            if self.mask[i] == False:
                continue
            rgb,_,_ = read_image(path, max_dim=max_dim)
            rgb = rgb.transpose([2, 0, 1])
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.object_masks = []
        for i, path in enumerate(mask_paths):
            if self.mask[i] == False:
                continue
            object_mask,_,_ = read_image(path, max_dim)
            self.object_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

        self.depths = []

        for i, path in enumerate(depth_paths):
            if self.mask[i] == False:
                continue
            depth = read_map(path, max_dim)
            depth = depth.transpose([2, 0, 1]).copy()
            self.depths.append(torch.from_numpy(depth).float())

        self.depth_range = torch.tensor([1., 3.])

    def __len__(self):
        return self.mask.sum()

    def __getitem__(self, idx):
        sample = {
            "object_mask": self.object_masks[idx],
            "intrinsics": self.intrinsics_all[idx],
            "depth": self.depths[idx],
            "extrinsics": self.extrinsics_all[idx]
        }

        ground_truth = {"rgb": self.rgb_images[idx]}
        ground_truth["rgb"] = self.rgb_images[idx]
        sample["object_mask"] = self.object_masks[idx]

        if not self.train_cameras:
            sample["c2w"] = self.c2w_all[idx]
        return idx, sample, ground_truth

    def get_all_imgs(self):
        all = torch.stack(self.rgb_images, dim=0)
        return all

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far
