import os
from tqdm import tqdm
import open3d as o3d
import numpy as np
import shutil
from points2depth import points2depth
from datasets.data_io import read_cam_file, save_map, save_image, read_map, read_image
from glob import glob
from PIL import Image
from natsort import natsorted
import cv2 as cv

def collate_points(src_path, tar_path):
    dirs = os.listdir(src_path)
    for dir in tqdm(dirs):
        print(f" processing {dir} ")
        src_dir = os.path.join(src_path, dir)
        tar_dir = os.path.join(tar_path, dir)
        if os.path.isdir(src_dir):
            os.makedirs(tar_dir, exist_ok=True)
            points_path1 = os.path.join(src_dir, "points.ply")
            if os.path.isfile(points_path1):
                # shutil.copy(points_path1, tar_dir)
                pcd = o3d.io.read_point_cloud(points_path1)
                points1 = np.asarray(pcd.points)
                ## align point cloud
                points1[..., 2] = points1[..., 2] + 4.5
                pcd.points = o3d.utility.Vector3dVector(points1)
                o3d.io.write_point_cloud(os.path.join(tar_dir, "points.ply"), pcd)


def delete_with_no_points(path):
    dirs = os.listdir(path)
    for dir in dirs:
        abs_dir = os.path.join(path, dir)
        points_path = os.path.join(abs_dir, "points.ply")
        if not os.path.exists(points_path) and os.path.isdir(abs_dir):
            shutil.rmtree(abs_dir)
            print(f" === delete {abs_dir} ===")

def check_if_scan_in_highres(path, lists="./lists/dtu/test.txt"):
    with open(lists) as f:
        scans = [line.rstrip() for line in f.readlines()]
    absent = []
    highres = os.listdir(path)
    # print(scans)
    # print(highres)
    for scan in scans:
        if scan not in highres:
            absent.append(scan)
    print(absent)

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


def gen_depths_and_masks(tar_path: str):
    # scan_lists = os.listdir(tar_path)
    scan_lists = sorted(os.listdir(tar_path))
    H, W = 512, 640
    for scan in scan_lists:
        print( f" === processing {scan} ===")
        abs_path = os.path.join(tar_path, scan)
        point_path = os.path.join(abs_path, "points.ply")
        depth_path = os.path.join(abs_path, "depth_gt")
        mask_path = os.path.join(abs_path, "masks")
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        ### read and preprocess point cloud ###
        pcd = o3d.io.read_point_cloud(point_path)
        pcd = pcd.voxel_down_sample(0.05)
        # pcd = pcd.remove_radius_outlier(4, 1.5)[0]
        points = np.asarray(pcd.points)
        camlist = os.listdir(os.path.join(abs_path, "cams"))
        for camfile in tqdm(camlist):
            intr, extr, depth_range = read_cam_file(os.path.join(abs_path, "cams", camfile))
            depth = points2depth(points, H, W, intr, extr)
            save_map(os.path.join(depth_path, camfile[:-8]+".pfm"), depth)
            mask = (depth > 1e-4)
            save_image(os.path.join(mask_path, camfile[:-8]+".png"), mask)

def png2jpg(tar_path):
    dirs = os.listdir(tar_path)
    for dir in dirs:
        print(f" == processing {dir} ==")
        abs_img_dir = os.path.join(tar_path, dir, "images")
        abs_img_list = glob(os.path.join(abs_img_dir, "*.png"))
        for img_path in tqdm(abs_img_list):
            image = np.array(Image.open(img_path), dtype=np.uint8)
            Image.fromarray(image).save(img_path[:-3]+"jpg")
            os.remove(img_path)

def convert_high_res_to_dtu_training(high_res_path, tar_path):
    scans = os.listdir(high_res_path)
    for scan in scans:
        print(f" == processing {scan} ==")
        if scan == "scan110":
            continue
        abs_depth_dir = os.path.join(tar_path, scan, "depth_gt")
        abs_mask_dir = os.path.join(tar_path, scan, "masks")

        os.makedirs(abs_depth_dir, exist_ok=True)
        os.makedirs(abs_mask_dir, exist_ok=True)
        abs_points_path = os.path.join(high_res_path, scan, "points.ply")
        tar_points_path = os.path.join(tar_path, scan)
        shutil.copy(abs_points_path, tar_points_path)

        for cam_file in os.listdir(os.path.join(tar_path, scan, "cams")):
            # Extract view ID and write cam file
            view_id = int(cam_file.split("_")[0])
            depth_map = read_map(os.path.join(
                high_res_path, scan, "depth_gt","{:0>8}.pfm".format(view_id)), 800)
            depth_map = depth_map[44:556, 80:720]
            save_map(os.path.join(tar_path, scan, "depth_gt", "{:0>8}.pfm".format(view_id)), depth_map)

            # Copy mask after resizing and cropping
            mask = read_image(os.path.join(
                high_res_path, scan, "masks", "{:0>8}.png".format(view_id)), 800)[0]

            mask = mask[44:556, 80:720] > 0.04
            mask = np.squeeze(mask)
            save_image(os.path.join(tar_path, scan, "masks", "{:0>8}.png".format(view_id)), mask)

def downsample_all_points(path):
    scans = os.listdir(path)
    for scan in tqdm(scans):
        point_path = os.path.join(path, scan, "points.ply")
        if os.path.exists(point_path):
            pcd = o3d.io.read_point_cloud(point_path)
            pcd = pcd.voxel_down_sample(0.68)
            pcd = pcd.remove_radius_outlier(4, 1.5)[0]
            o3d.io.write_point_cloud(os.path.join(path,scan,"points_voxel_downsampled2.ply"), pcd)

def downsample_all_stlpoints(path):
    pts_files = os.listdir(path)
    for f in tqdm(pts_files):
        file = os.path.join(path, f)
        pcd = o3d.io.read_point_cloud(file)
        print("before pts num ", np.asarray(pcd.points).shape)
        pcd = pcd.voxel_down_sample(0.8)
        pcd = pcd.remove_radius_outlier(4, 1.5)[0]
        print("after pts num ", np.asarray(pcd.points).shape)
        o3d.io.write_point_cloud(file, pcd)


def collate_points_in_a_file(src_path, tar_path):
    dirs = os.listdir(src_path)
    for dir in tqdm(dirs):
        print(f" processing {dir} ")
        src_dir = os.path.join(src_path, dir)
        os.makedirs(tar_path, exist_ok=True)

        if os.path.isdir(src_dir):
            points_path1 = os.path.join(src_dir, "points.ply")
            if os.path.isfile(points_path1):
                scan_number = int(dir.replace("scan", ""))
                tar_file = os.path.join(tar_path, "stl{:0>3}_total.ply".format(scan_number))
                shutil.copyfile(points_path1, tar_file)

def point_ras(scene_path, out_path):
    pcd = o3d.io.read_point_cloud(os.path.join(scene_path, "points_voxel_downsampled.ply"))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    camlist = sorted(os.listdir(os.path.join(scene_path, "cams")))
    os.makedirs(out_path, exist_ok=True)
    for i, camfile in enumerate(tqdm(camlist)):
        intr, extr, depth_range = read_cam_file(os.path.join(scene_path, "cams", camfile))
        point_w = np.concatenate([points, np.ones_like(points)[..., :1]], axis=-1)
        point_c = point_w @ extr.T
        point_c = point_c[..., :3]
        point_i = (point_c @ intr.T)
        point_i = point_i[..., :2] / point_i[..., 2:]
        # print(point_i[..., 0].max(), "   ", point_i[..., 0].min(), "   ", point_i[..., 1].max(), "   ", point_i[..., 1].min())
        u, v = np.round(point_i[..., 0]).astype(np.int), np.round(point_i[..., 1]).astype(np.int)
        # print(u.max(), "   ", u.min(), "   ", v.max(), "   ", v.min())
        z = point_c[..., 2]
        valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < 640)), np.bitwise_and((v >= 0), (v < 512)))

        u, v, z, c = u[valid], v[valid], z[valid], colors[valid]
        img = np.full((512, 640, 3), 0, dtype=np.float32)
        img_z = np.full((512, 640), 10000, dtype=np.float32)
        for ui, vi, zi, ci in zip(u, v, z, c):
            if zi <= img_z[vi, ui]:
                # print(vi, "  ", ui)
                img_z[vi, ui] = zi
                img[vi, ui, 0] = ci[0]
                img[vi, ui, 1] = ci[1]
                img[vi, ui, 2] = ci[2]

        # img_z = np.min(img_z_shift, axis=0)
        img_z[img_z == 10000] = 0
        mask = (img_z > 0)
        save_image(os.path.join(out_path, f"image_{i:03d}.png"), img)
        save_image(os.path.join(out_path, f"mask_{i:03d}.png"), mask)

def copy2dest(src_path, tar_path):
    os.makedirs(tar_path, exist_ok=True)

    scans = os.listdir(src_path)
    scans = natsorted(scans)

    dirs_need_copy = ["cams", "depth_gt", "images", "masks"]
    files_need_copy = ['points_voxel_downsampled.ply', 'pair.txt']
    for scan in scans:
        print(f" ====== in processing {scan} ========= ")
        old_scan_dir = os.path.join(src_path, scan)
        new_scan_dir = os.path.join(tar_path, scan)
        os.makedirs(new_scan_dir, exist_ok=True)
        try:
            for f in files_need_copy:
                shutil.copy(os.path.join(old_scan_dir, f), os.path.join(new_scan_dir, f))
        except:
            continue
        for dir in dirs_need_copy:
            shutil.copytree(os.path.join(old_scan_dir, dir), os.path.join(new_scan_dir, dir))
        
