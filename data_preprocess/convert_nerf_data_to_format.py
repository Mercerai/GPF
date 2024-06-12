import os
import numpy
import shutil
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import open3d as o3d
from points2depth import points2depth
from datasets.data_io import read_cam_file, save_map, save_image, read_map, read_image

def collate_points_and_imgs(meshes_path, data_path):
    scene_list = os.path.join(data_path, "training_scene.txt")
    if os.path.isfile(scene_list):
        with open(scene_list) as f:
            scenes = [line.rstrip() for line in f.readlines()]

    for scene in scenes:
        print(f" =================== processing ======================= {scene}")
        mesh_path = os.path.join(meshes_path, scene + ".ply")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        points = mesh.sample_points_uniformly(15000000)
        point_path = os.path.join(data_path, scene, "points.ply")
        o3d.io.write_point_cloud(point_path, points)

        # image_path = os.path.join(data_path, scene, "images")
        # test_path = os.path.join(data_path, scene, "test")
        # os.makedirs(image_path, exist_ok=True)
        #
        # for i in range(200):
        #     src_img_path = os.path.join(test_path, "r_{:d}.png".format(i))
        #     shutil.copy(src_img_path, image_path)

def convert_image_alpha_to_image(data_path):
    scene_list = os.path.join(data_path, "training_scene.txt")
    if os.path.isfile(scene_list):
        with open(scene_list) as f:
            scenes = [line.rstrip() for line in f.readlines()]

    for scene in scenes:
        print(f" =================== processing ======================= {scene}")
        image_path = os.path.join(data_path, scene, "images")
        test_path = os.path.join(data_path, scene, "test")
        os.makedirs(image_path, exist_ok=True)

        for i in tqdm(range(200)):
            src_img_path = os.path.join(test_path, "r_{:d}.png".format(i))
            img = Image.open(src_img_path)
            img = np.array(img, dtype=np.uint8)[..., :3]
            Image.fromarray(img).save(os.path.join(image_path, "r_{:d}.png".format(i)))

def gen_depths_and_masks(data_path):
    scene_list = os.path.join(data_path, "training_scene.txt")
    if os.path.isfile(scene_list):
        with open(scene_list) as f:
            scenes = [line.rstrip() for line in f.readlines()]

    H, W = 800, 800
    for scene in scenes:
        print(f"process {scene}")
        depth_folder = os.path.join(data_path, scene, "depth_gt")
        mask_folder = os.path.join(data_path, scene, "masks")
        cam_file = os.path.join(data_path, scene, "transforms_test.json")

        os.makedirs(depth_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)

        points = o3d.io.read_point_cloud(os.path.join(data_path, scene, "points.ply"))
        points = np.asarray(points.points)
        with open(cam_file) as fp:
            meta = json.load(fp)
        camera_angle_x = float(meta["camera_angle_x"])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

        for idx, frame in enumerate(tqdm(meta["frames"])):
            c2w = np.array(frame["transform_matrix"]).astype(np.float32)
            extrinsics = np.linalg.inv(c2w)
            depth = points2depth(points, H, W, K, extrinsics, nerf_flag=True)
            save_map(os.path.join(depth_folder, "r_{:d}.pfm".format(idx)), depth)
            mask = (depth > 1e-5)
            save_image(os.path.join(mask_folder, "r_{:d}.png".format(idx)), mask)


if __name__ == "__main__":
    collate_points_and_imgs(r"/home/share/jx/nerf_ply/", r"/home/share/jx/nerf_synthetic_high_density_points/")
    gen_depths_and_masks(r"/home/share/jx/nerf_synthetic_high_density_points/")
    # convert_image_alpha_to_image(r"/home/share/jx/nerf_synthetic/")

