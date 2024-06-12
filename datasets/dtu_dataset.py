import numpy as np
import os
import random
import torch
from .data_io import read_cam_file, read_image, read_map, read_pair_file, read_ply
from torch.utils.data import Dataset
from typing import List, Tuple

class DTUDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            point_path,
            num_views: int = 10,
            max_dim: int = -1,
            scan_list: str = '',
            num_light_idx: int = -1,
            cam_folder: str = "cams",
            pair_path: str = "pair.txt",
            image_folder: str = "images",
            depth_folder: str = "depth_gt",
            mask_folder: str = "masks",
            image_extension: str = ".jpg",
            robust_train: bool = False,
            specific_light_idx = -1,
            pcd_downsample = 0,
            downsample_method = "uniform"
    ) -> None:

        super(DTUDataset, self).__init__()

        self.data_path = data_path
        self.pair_path = pair_path
        self.point_path = point_path
        self.num_views = num_views
        self.max_dim = max_dim
        self.robust_train = robust_train
        self.cam_folder = cam_folder
        self.depth_folder = depth_folder
        self.mask_folder = mask_folder
        self.image_folder = image_folder
        self.image_extension = image_extension
        self.pcd_downsample = pcd_downsample
        self.downsample_method = downsample_method
        self.metas: List[Tuple[str, str, int, List[int]]] = []

        if os.path.isfile(scan_list):
            with open(scan_list) as f:
                scans = [line.rstrip() for line in f.readlines()]
        else:
            scans = ['']

        if num_light_idx > 0:
            light_indexes = [str(idx) for idx in range(num_light_idx)]
        else:
            light_indexes = ['']

        if specific_light_idx > 0:
            light_indexes = [int(specific_light_idx)]

        for scan in scans:
            pair_data = read_pair_file(os.path.join(self.data_path, scan, pair_path))
            for light_idx in light_indexes:
                self.metas += [(scan, light_idx, ref, src) for ref, src in pair_data]
        # print("========= Dataset initialization Done ==================")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first num_views source views
        num_src_views = min(len(src_views), self.num_views)
        scan_number = scan.replace("scan", "")
        if (self.point_path is not None) and (self.point_path != "None"):
            points_filename = os.path.join(self.point_path , "stl{:0>3}_total.ply".format(scan_number))
        else:

            # points_filename = os.path.join(self.data_path, scan, "points_uniform_downsampled.ply")
            points_filename = os.path.join(self.data_path, scan, "points_voxel_downsampled.ply")

        points = read_ply(points_filename, self.pcd_downsample, self.downsample_method)

        all = [ref_view] + src_views

        if self.robust_train:
            random.shuffle(all)
        all = [all[0]] + all[1:num_src_views+1]
        view_ids = all

        #### read all cam files , we use distance-sort to replace the pair file to instruct source images selection##
        c2ws = []
        for idx in range(49):
            cam_filename = os.path.join(self.data_path, scan, self.cam_folder, "{:0>8}_cam.txt".format(idx))
            intrinsic, extrinsic, depth_params = read_cam_file(cam_filename)
            c2ws.append(np.linalg.inv(extrinsic))

        cam_points = np.array([c2w[:3, 3] for c2w in c2ws])
        cam_point = c2ws[ref_view][:3, 3]
        distance = np.linalg.norm(cam_points - cam_point[np.newaxis], axis=-1)
        argsorts = distance.argsort()
        argsorts = argsorts[1:]
        src_imgs_idx = argsorts[:num_src_views].tolist()
        view_ids = [ref_view] + src_imgs_idx

        images = []
        intrinsics = []
        extrinsics = []
        depths = []

        for view_index, view_id in enumerate(view_ids):
            img_filename = os.path.join(
                self.data_path, scan, self.image_folder, light_idx, "{:0>8}{}".format(view_id, self.image_extension))

            ### cover image noises via mask ##
            mask_filename = os.path.join(
                self.data_path, scan, self.mask_folder, "{:0>8}{}".format(view_id, ".png"))

            the_mask, _, _ = read_image(mask_filename)
            the_mask = the_mask[:,:,np.newaxis]

            image, original_h, original_w = read_image(img_filename, self.max_dim)
            image = image * the_mask
            images.append(image.transpose([2, 0, 1]))

            cam_filename = os.path.join(self.data_path, scan, self.cam_folder, "{:0>8}_cam.txt".format(view_id))

            intrinsic, extrinsic, depth_params = read_cam_file(cam_filename)

            intrinsic[0] *= image.shape[1] / original_w
            intrinsic[1] *= image.shape[0] / original_h
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            depth_gt_filename = os.path.join(self.data_path, scan, self.depth_folder, "{:0>8}.pfm".format(view_id))
            dpt = read_map(depth_gt_filename, self.max_dim).transpose([2, 0, 1]).copy()
            # print(dpt.shape)
            depths.append(dpt)

            if view_index == 0:  # reference view
                if depth_params[1] < 3:
                    depth_params[1] = 935.0

                depth_min = depth_params[0]
                depth_max = depth_params[1]
                # Using `copy()` here to avoid the negative stride resulting from the transpose
                # Create mask from GT depth map
                depth_range = depth_params
                ref_mask = (dpt >= depth_min)

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)

        return {
            # "view_ids": view_ids,
            "images": images,          # List[Tensor]: [N][3,Hi,Wi], N is number of images where N is num of views
            "points": points, # Tensor: [N, 3]
            "intrinsics": intrinsics,  # Tensor: [N,3,3]
            "extrinsics": extrinsics,  # Tensor: [N,4,4]
            "depth_range": depth_range, #Tensor [2]
            "depths_gt": depths,      # Tensor: [1,H0,W0] if exists
            "ref_mask": ref_mask,              # Tensor: [1,H0,W0] if exists
            # "filename": os.path.join(scan, "{}", "{:0>8}".format(view_ids[0]) + "{}")
            "filename": os.path.join(scan, f"{light_idx}", "{:0>8}.png".format(view_ids[0]))
        }

    def get_single_(self, scan_num, light_idx, ref_view, target_c2w=None):
        """
        return Tensor of one single training sample
        :param scan_num:
        :param light_idx:
        :param ref_view:
        :return:
        """
        the_scan = "scan"+str(scan_num)
        pair_data = read_pair_file(os.path.join(self.data_path, the_scan, self.pair_path))
        ref_view, src_views = pair_data[ref_view]

        if (self.point_path is not None) and (self.point_path != "None"):
            points_filename = os.path.join(self.point_path, "stl{:0>3}_total.ply".format(scan_num))
        else:
            # points_filename = os.path.join(self.data_path, "scan{}".format(scan_num), "points_uniform_downsampled.ply")
            points_filename = os.path.join(self.data_path, "scan{}".format(scan_num), "points_voxel_downsampled.ply")

        points = read_ply(points_filename, self.pcd_downsample, self.downsample_method)
        points = torch.from_numpy(points).unsqueeze(0)
        # print(" points shape ",points.shape)

        if light_idx == -1:
            light_idx = ""

        view_ids = [ref_view] + src_views

        #### read all cam files , we use distance-sort to replace the pair file to instruct source images selection##
        c2ws = []
        for idx in range(49):
            cam_filename = os.path.join(self.data_path, the_scan, self.cam_folder, "{:0>8}_cam.txt".format(idx))
            intrinsic, extrinsic, depth_params = read_cam_file(cam_filename)
            c2ws.append(np.linalg.inv(extrinsic))

        cam_points = np.array([c2w[:3, 3] for c2w in c2ws])
        if target_c2w is not None:
            cam_point = target_c2w[:3, 3]
        else:
            cam_point = c2ws[ref_view][:3, 3]

        distance = np.linalg.norm(cam_points - cam_point[np.newaxis], axis=-1)
        argsorts = distance.argsort()
        if target_c2w is None:
            argsorts = argsorts[1:]

        src_imgs_idx = argsorts[:self.num_views].tolist()
        view_ids = [ref_view] + src_imgs_idx

        scan = "scan" + str(scan_num)
        light_idx = str(light_idx)
        images = []
        intrinsics = []
        extrinsics = []
        depths = []

        for view_index, view_id in enumerate(view_ids):
            img_filename = os.path.join(self.data_path, scan, self.image_folder, light_idx, "{:0>8}{}".format(view_id, self.image_extension))

            mask_filename = os.path.join(
                self.data_path, scan, self.mask_folder, "{:0>8}{}".format(view_id, ".png"))

            the_mask, _, _ = read_image(mask_filename)
            the_mask = the_mask[:, :, np.newaxis]

            image, original_h, original_w = read_image(img_filename, self.max_dim)
            image = image * the_mask

            image_to_add_in_list = image.transpose([2, 0, 1])
            image_to_add_in_list = torch.from_numpy(image_to_add_in_list)[None]
            images.append(image_to_add_in_list)
            cam_filename = os.path.join(self.data_path, scan, self.cam_folder, "{:0>8}_cam.txt".format(view_id))

            intrinsic, extrinsic, depth_params = read_cam_file(cam_filename)

            intrinsic[0] *= image.shape[1] / original_w
            intrinsic[1] *= image.shape[0] / original_h
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            depth_gt_filename = os.path.join(self.data_path, scan, self.depth_folder, "{:0>8}.pfm".format(view_id))
            dpt = read_map(depth_gt_filename, self.max_dim).transpose([2, 0, 1]).copy()
            dpt = torch.from_numpy(dpt)[None] #[1,C,H,W]
            depths.append(dpt)

            if view_index == 0:  # reference view
                if depth_params[1] < 3:
                    depth_params[1] = 935.0

                depth_min = depth_params[0]
                depth_max = depth_params[1]
                # Using `copy()` here to avoid the negative stride resulting from the transpose
                # Create mask from GT depth map
                depth_range = torch.from_numpy(depth_params)[None]
                ref_mask = (dpt >= depth_min)

        intrinsics = np.stack(intrinsics) #[S, 3, 3]
        extrinsics = np.stack(extrinsics) #[S, 4, 4]
        intrinsics, extrinsics = torch.from_numpy(intrinsics)[None], torch.from_numpy(extrinsics)[None]

        return {
            # "view_ids": view_ids,
            "images": images,  # List[Tensor]: [N][1, 3,Hi,Wi], N is number of images where N is num of views
            "points": points,  # Tensor: [1, N, 3]
            "intrinsics": intrinsics,  # Tensor: [1, S ,3,3]
            "extrinsics": extrinsics,  # Tensor: [1,S ,4,4]
            "depth_range": depth_range,  # Tensor [1, 2]
            "depths_gt": depths,  # Tensor: [1, 1, H0, W0] if exists
            "ref_mask": ref_mask,  # Tensor: [1, 1,H0,W0] if exists
            # "filename": os.path.join(scan, "{}", "{:0>8}".format(view_ids[0]) + "{}")
            "filename": [os.path.join(scan, f"{light_idx}", "{:0>8}.png".format(view_ids[0]))]
        }


if __name__ == "__main__":
    import torch
    dataset = DTUDataset(data_path=r"/home/share/jx/dtu_training/dtu/",
                         point_path=r"/home/share/jx/Points/stl/",
                         num_views=5,
                         num_light_idx=7,
                         scan_list=r"../lists/dtu/all.txt",
                         specific_light_idx=-1)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    print("len(dataloader) : ",len(dataloader))
    dataloader.dataset.get_single_(1,3,0)
    for b in dataloader:
        # print(b)
        for k, v in b.items():
            print(k)
            print(type(v))
            if isinstance(v, torch.Tensor):
                print("shape is ", v.shape)
            elif isinstance(v, list):
                # print("list content is ", v)
                print(len(v))
                if isinstance(v[0], str):
                    print(v[0])
                else:
                    print(v[0].shape)
            print("")
        print(" =========================== batch end =======================")
        break
