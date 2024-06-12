import os
import imageio
import cv2
import torch
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import numpy as np
from datasets.data_io import save_map
from tqdm import tqdm
from utils import rend_utils
from utils.utils import compute_loss, adjust_learning_rate
from models.point_volumetric_renderer import Point_Volumetric_Renderer
from datasets.dtu_dataset import DTUDataset
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from utils import config_utils
from points2depth import points2depth_tensor

DEBUG=False

if DEBUG:
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    torch.autograd.set_detect_anomaly(True)

# torch.autograd.set_detect_anomaly(True)
# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def to_cuda(element):
    pass

def integerify(img):
    return (img * 255.0).astype(np.uint8)



def interpolate_pose_between(pose_0, pose_1, ratio):
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)

    return pose

def gen_interpolate_poses(pose_0, pose_1, n_views=60):
    poses_all = []
    for i in range(n_views):
        poses_all.append(interpolate_pose_between(pose_0, pose_1, np.sin(((i / n_views) - 0.5) * np.pi) * 0.5 + 0.5))
    return poses_all

def render_path(args, model, test_loader):
    device = args.device
    os.makedirs("./log", exist_ok=True)
    test_dir = os.path.join("log", args.expname, "render_path")
    os.makedirs(test_dir, exist_ok=True)

    H, W = test_loader.dataset[0]["images"][0].shape[1:]
    pose_0 = np.linalg.inv(test_loader.dataset[args.interp_0]["extrinsics"][0])
    pose_1 = np.linalg.inv(test_loader.dataset[args.interp_1]["extrinsics"][0])

    poses_all = gen_interpolate_poses(pose_0, pose_1, args.num_views)
    images = []
    with torch.no_grad():
        for i, pose in enumerate(tqdm(poses_all)):
            batch = test_loader.dataset.get_single_(args.scan_num, 3, args.view_num, pose)
            imgs, points, intrinsics, extrinsics, dpts, msk, depth_range = \
                batch["images"], batch["points"].to(device), batch["intrinsics"].to(device), batch["extrinsics"].to(
                    device), batch["depths_gt"], batch["ref_mask"].to(device), batch["depth_range"].to(device)
            depth = points2depth_tensor(points.to(device), H, W, intrinsics[0, 0].to(device), torch.inverse(torch.from_numpy(pose)).to(device))
            depth = depth.to(device)
            
            tgt_img = imgs[0].to(device)  # [B(1), 3, H, W]
            src_imgs = torch.stack(imgs[1:], dim=1).to(device)  # B,S,C,H,W
            dpts = torch.stack(dpts, dim=1).to(device)  # B,S+1, C,H,W
            tgt_c2w = torch.linalg.inv(extrinsics[0, 0])[None]  # [1, 4,4]
            tgt_c2w = torch.from_numpy(pose)[None].to(device)

            rays_o, rays_d, rgb_gt, selected_inds = rend_utils.build_rays(intrinsics[:, 0, :, :], tgt_c2w,
                                                                          tgt_img, msk, N_rays=-1,
                                                                          sample_on_mask=False)

            if args.sample_strategy.test == "uniform":
                pts_info = rend_utils.uniform_sampling(args, rays_o, rays_d, depth_range)
            elif args.sample_strategy.test == "log":
                pts_info = rend_utils.log_sampling(args, rays_o, rays_d, depth[None], selected_inds, depth_range)

            pts_info["extrinsics"] = extrinsics[:, 0]
            pts_info["depth"] = dpts[:, 0, :, selected_inds[..., :, 0], selected_inds[..., :, 1]]
            model.set_points(points)

            ret = model(src_imgs, pts_info, intrinsics[:, 1:, :, :], extrinsics[:, 1:, :, :], dpts[:, 1:])
            rgb, depth, acc = ret["rgb"], ret["depth"], ret["acc_map"]
            rgb = rgb.data.cpu().numpy().reshape(H, W, 3)
            msk = msk.data.cpu().numpy().reshape(H, W, 1)
            # rgb = rgb * msk
            img = integerify(rgb)

            img[..., [0, 2]] = img[..., [2, 0]]

            depth = depth.data.cpu().numpy().reshape(H, W)
            acc = acc.data.cpu().numpy().reshape(H, W)

            filename = batch["filename"][0]

            os.makedirs(os.path.join(test_dir, os.path.dirname(filename)), exist_ok=True)

            name, ext = os.path.splitext(os.path.basename(filename))
            imgname = "{:0>6d}_".format(i) + name + "_img" + ext
            depthname = "{:0>6d}_".format(i) + name + "_depth" + ext
            accname = "{:0>6d}_".format(i) + name + "_acc" + ext
            tgt_img = tgt_img.squeeze().permute(1, 2, 0).data.cpu().numpy()
            tgt_img = integerify(tgt_img)
            tgt_img[..., [0, 2]] = tgt_img[..., [2, 0]]

            cv2.imwrite(
                os.path.join(test_dir, os.path.dirname(filename), imgname),
                np.concatenate([img], axis=0)
            )
            # cv2 is bgr image rather than rgb
            # img[..., [0, 2]] = img[..., [2, 0]]
            images.append(img)
            # tgt_depth = dpts[:, 0].squeeze().data.cpu().numpy()

            # cv2.imwrite(
            #     os.path.join(test_dir, os.path.dirname(filename), depthname),
            #     np.concatenate([integerify(depth / depth.max()), integerify(tgt_depth / tgt_depth.max())], axis=0)
            # )
            #
            # cv2.imwrite(
            #     os.path.join(test_dir, os.path.dirname(filename), accname),
            #     integerify(acc)
            # )
        n_frames= len(images)
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(os.path.join(test_dir, os.path.dirname(filename), "img_video.avi"),
                            fourcc, 2, (W, H))

    for image in images:
        writer.write(image)
    writer.release()

    os.makedirs("./log", exist_ok=True)
    output_dir = os.path.join("log", args.expname)
    os.makedirs(output_dir, exist_ok=True)

    device = args.device

    n_epoch = args.training.num_epoch

    H, W = dataloader.dataset[0]["images"][0].shape[1:]
    print(f"========= H and W are : {H} X {W} ===========")

    os.makedirs(os.path.join(output_dir, "tflogs"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tflogs"))
    n_batch = len(dataloader)

    start_epoch = int(global_step // n_batch)
    for epoch in range(start_epoch, n_epoch):
        for iter, batch in enumerate(tqdm(dataloader, desc=f"{args.expname} epoch:{epoch}")): #
    #list[n]:(B(1),3,H,W) #(1,N,3) (1,n,3,3) (1,n,4,4) n = n_views
            imgs, points, intrinsics, extrinsics, dpts, msk, depth_range = \
                batch["images"], batch["points"].to(device), batch["intrinsics"].to(device), batch["extrinsics"].to(device), \
                batch["depths_gt"], batch["ref_mask"].to(device), batch["depth_range"].to(device)
            tgt_img = imgs[0].to(device) #[B(1), 3, H, W]
            src_imgs = torch.stack(imgs[1:], dim=1).to(device) #B,S,C,H,W
            dpts = torch.stack(dpts, dim=1).to(device) #B,S+1, C,H,W
            tgt_c2w = torch.inverse(extrinsics[:, 0]) #[1,4,4]
            rays_o, rays_d, rgb_gt, selected_inds = rend_utils.build_rays(intrinsics[:,0, :, :],
                                                            tgt_c2w, tgt_img, msk,
                                                            N_rays=args.training.N_rays,
                                                            sample_on_mask=args.training.sample_on_mask) # B, N_rays, 3 3 3 2
            if args.sample_strategy.train == "surface":
                pts_info = rend_utils.surface_sampling(args, rays_o, rays_d, dpts[:, 0], selected_inds, depth_range)
            elif args.sample_strategy.train == "uniform":
                pts_info = rend_utils.uniform_sampling(args, rays_o, rays_d, depth_range)
            elif args.sample_strategy.train == "log":
                pts_info = rend_utils.log_sampling(args, rays_o, rays_d, dpts[:, 0], selected_inds, depth_range)
            else:
                raise NotImplementedError

            pts_info["extrinsics"] = extrinsics[:, 0]
            pts_info["depth"] = dpts[:, 0, :, selected_inds[..., :, 0], selected_inds[..., :, 1]]

            model.set_points(points)

            ret = model(src_imgs, pts_info, intrinsics[:, 1:, :, :], extrinsics[:, 1:, :, :], dpts[:,1:])
            loss_dict = compute_loss(ret, rgb_gt, dpts[:, 0, :, selected_inds[..., :, 0], selected_inds[..., :, 1]],
                                     msk[..., selected_inds[..., :, 0], selected_inds[..., :, 1]], pts_info["z_vals"], global_step)
            loss = loss_dict["rgb loss"] + args.training.w_m * loss_dict["mask loss"] + args.training.w_d * loss_dict["depth loss"] + args.training.w_s*loss_dict["weights loss"]
            # loss = loss_dict["rgb loss"] + args.training.w_m * loss_dict["mask loss"] + args.training.w_d* loss_dict["depth loss"]
            writer.add_scalar('Loss/loss', loss, global_step+1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            # adjust_learning_rate(optimizer, args.training.lr, global_step, warm_up_end=1000, max_step=200000)

            # model.release_neural_pts_cache()
            if iter % args.training.print_interval == 0 or iter == (n_batch - 1):
                mask_loss = loss_dict["mask loss"]
                rgb_loss = loss_dict["rgb loss"]
                depth_loss = loss_dict["depth loss"]
                weights_loss = loss_dict["weights loss"]
                sparse_loss = loss_dict["sparse loss"]
                print(f"Epoch {epoch}  Batch {iter} loss {loss.item()}  rgb loss {rgb_loss} mask loss"
                      f" {mask_loss} depth loss {depth_loss} weights loss {weights_loss} sparse loss {sparse_loss}")
                print("lr : ", optimizer.param_groups[0]['lr'])

            if global_step % args.training.i_test_img == 0 and global_step != 0:
                # model.eval()
                test(args, model, test_loader, global_step, specific_idx=args.training.test_specific)
                test(args, model, test_loader, global_step, specific_idx=(114, 3, 35))
                test(args, model, test_loader, global_step, specific_idx=(114, 3, 32))

                model.neural_points.release_cache()

            if global_step % 10000 == 0 and global_step != 0:
                tlst = [1, 4, 9 ,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,118]
                for i in tlst:
                    test(args, model, test_loader, global_step, specific_idx=(int(i), 3, 32))

        if (epoch + 1) % args.training.save_ep == 0:
            path = os.path.join(output_dir, 'ckpts')
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, 'e{:03d}_step{:06d}.tar'.format(epoch, global_step))
            torch.save({
                "global_step": global_step,
                "feature_net_dict": model.feature_net.state_dict(),
                "neural_points_dict": model.neural_points.state_dict(),
                "optimizer_dict": optimizer.state_dict()}, file_path)

            print("saved checkpoint at :", file_path)

def test(args, model, test_loader, global_step, specific_idx=None):
    device = args.device
    os.makedirs("./log", exist_ok=True)
    test_dir = os.path.join("log", args.expname, "evaluation")
    os.makedirs(test_dir, exist_ok=True)

    H, W = test_loader.dataset[0]["images"][0].shape[1:]

    with torch.no_grad():
        if specific_idx is not None:
            assert len(specific_idx) == 3, "specific_idx ought to be (scanid, lightid, refid)"
            batch = test_loader.dataset.get_single_(specific_idx[0], specific_idx[1], specific_idx[2])

            imgs, points, intrinsics, extrinsics, dpts, msk, depth_range = \
                batch["images"], batch["points"].to(device), batch["intrinsics"].to(device), batch["extrinsics"].to(
                    device), batch["depths_gt"], batch["ref_mask"].to(device), batch["depth_range"].to(device)

            tgt_img = imgs[0].to(device)  # [B(1), 3, H, W]
            src_imgs = torch.stack(imgs[1:], dim=1).to(device)  # B,S,C,H,W
            dpts = torch.stack(dpts, dim=1).to(device)  # B,S+1, C,H,W
            tgt_c2w = torch.linalg.inv(extrinsics[0, 0])[None]  # [1, 4,4]

            rays_o, rays_d, rgb_gt, selected_inds = rend_utils.build_rays(intrinsics[:, 0, :, :], tgt_c2w,
                                                                          tgt_img, msk, N_rays=-1,
                                                                          sample_on_mask=False)

            if args.sample_strategy.test == "surface":
                pts_info = rend_utils.surface_sampling(args, rays_o, rays_d, dpts[:, 0], selected_inds, depth_range)
            elif args.sample_strategy.test == "uniform":
                pts_info = rend_utils.uniform_sampling(args, rays_o, rays_d, depth_range)
            elif args.sample_strategy.train == "log":
                pts_info = rend_utils.log_sampling(args, rays_o, rays_d, dpts[:, 0], selected_inds, depth_range)
            else:
                raise NotImplementedError

            pts_info["extrinsics"] = extrinsics[:, 0]
            pts_info["depth"] = dpts[:, 0, :, selected_inds[..., :, 0], selected_inds[..., :, 1]]
            model.set_points(points)

            ret = model(src_imgs, pts_info, intrinsics[:, 1:, :, :], extrinsics[:, 1:, :, :], dpts[:, 1:])
            rgb, depth, acc = ret["rgb"], ret["depth"], ret["acc_map"]
            rgb = rgb.data.cpu().numpy().reshape(H, W, 3)
            msk = msk.data.cpu().numpy().reshape(H, W, 1)
            # rgb = rgb * msk
            img = integerify(rgb)
            img[..., [0, 2]] = img[..., [2, 0]]

            depth = depth.data.cpu().numpy().reshape(H, W)
            acc = acc.data.cpu().numpy().reshape(H, W)

            filename = batch["filename"][0]

            os.makedirs(os.path.join(test_dir, os.path.dirname(filename)), exist_ok=True)

            name, ext = os.path.splitext(os.path.basename(filename))
            imgname = "{:0>6d}_".format(global_step)+name+"_img"+ext
            depthname = "{:0>6d}_".format(global_step)+name+"_depth"+ext
            accname = "{:0>6d}_".format(global_step)+name+"_acc"+ext
            tgt_img = tgt_img.squeeze().permute(1,2,0).data.cpu().numpy()
            tgt_img = integerify(tgt_img)
            tgt_img[..., [0, 2]] = tgt_img[..., [2, 0]]

            cv2.imwrite(
                os.path.join(test_dir, os.path.dirname(filename), imgname),
                np.concatenate([img, tgt_img], axis=0)
            )

            tgt_depth = dpts[:, 0].squeeze().data.cpu().numpy()

            cv2.imwrite(
                os.path.join(test_dir, os.path.dirname(filename), depthname),
                np.concatenate([integerify(depth / depth.max()), integerify(tgt_depth / tgt_depth.max())], axis=0)
            )

            cv2.imwrite(
                os.path.join(test_dir, os.path.dirname(filename), accname),
                integerify(acc)
            )
            img[..., [0, 2]] = img[..., [2, 0]]
            ret_depth = integerify(depth / depth.max())
            return img, ret_depth
        else:
            for batch in test_loader:
                imgs, points, intrinsics, extrinsics, dpts, msk, depth_range = \
                    batch["images"], batch["points"].to(device), batch["intrinsics"].to(device), batch["extrinsics"].to(device), \
                    batch["depths_gt"], batch["ref_mask"].to(device), batch["depth_range"].to(device)

                tgt_img = imgs[0].to(device)  # [B(1), 3, H, W]
                src_imgs = torch.stack(imgs[1:], dim=1).to(device)  # B,S,C,H,W
                dpts = torch.stack(dpts, dim=1).to(device)  # B,S+1, C,H,W
                tgt_c2w = torch.inverse(extrinsics[:, 0])  # [1,4,4]
                rays_o, rays_d, rgb_gt, selected_inds = rend_utils.build_rays(intrinsics[:, 0, :, :], tgt_c2w, tgt_img,
                                                                              msk, N_rays=-1,sample_on_mask=None)  # B, N_rays, (3 3 3 2)

                #
                if args.sample_strategy == "surface":
                    pts_info = rend_utils.surface_sampling(args, rays_o, rays_d, dpts[:, 0], selected_inds, depth_range)
                elif args.sample_strategy == "uniform":
                    pts_info = rend_utils.uniform_sampling(args, rays_o, rays_d, depth_range)
                else:
                    raise NotImplementedError

                model.set_points(points)
                ret = model(src_imgs, pts_info, intrinsics[:, 1:, :, :], extrinsics[:, 1:, :, :], dpts[:, 1:])
                rgb, depth = ret["rgb_map"], ret["depth_map"]
                rgb = rgb.data.cpu().numpy().reshape(H, W, 3)
                img = integerify(rgb)
                img[..., [0, 2]] = img[..., [2, 0]]

                filename = batch["filename"][0]
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                cv2.imwrite(
                    os.path.join(test_dir, filename),
                    img
                )
                cv2.imwrite(
                    os.path.join(os.path.dirname(filename), "depth_", os.path.basename(filename)),
                    integerify(depth)
                )

def main(args):
    renderer = Point_Volumetric_Renderer(args, args.model.K, args.model.radius, args.model.n_valid_src_imgs)

    optimizer = torch.optim.Adam(renderer.parameters(), lr=args.training.lr)
    global_step = 0

    if args.checkpoint:
        ckpt_file = args.checkpoint
        print("=> Use ckpt:" + str(ckpt_file))
        state_dict = torch.load(ckpt_file, map_location=args.device)
        renderer.feature_net.load_state_dict(state_dict["feature_net_dict"])
        renderer.neural_points.load_state_dict(state_dict["neural_points_dict"])
        optimizer.load_state_dict(state_dict["optimizer_dict"])
        global_step = state_dict["global_step"]

    renderer.to(args.device)

    test_dataset = DTUDataset(
        # data_path=args.data.data_path,
        # point_path=args.data.point_path,
        data_path=args.data.test_data_path,
        point_path = args.data.point_path,
        num_views=args.data.num_src_views,
        scan_list=args.data.test_scan_list,
        robust_train=False,
        num_light_idx=7,
        # specific_light_idx=3,
        cam_folder=args.data.cam_folder,
        image_folder=args.data.image_folder,
        depth_folder=args.data.depth_folder,
        image_extension=args.data.image_extension,
        # pcd_downsample=args.data.pcd_downsample,
        # downsample_method=args.data.downsample_method
        pcd_downsample= 0.5,
        downsample_method="voxel"
    )

    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.data.batch_size, num_workers=args.data.num_workers, shuffle=False)

    # test(args, renderer, test_loader, global_step, specific_idx=(114, 3, 35))
    if not args.render_path:
        test(args, renderer, test_loader, global_step, specific_idx=(args.scan_num, 3, args.view_num))
    else:
        render_path(args, renderer, test_loader)
    # images = []
    # depths = []
    # for i in range(49):
    #     img, depth  = test(args, renderer, test_loader, global_step, specific_idx=(114, 3, int(i)))
    #     images.append(img)
    #     depths.append(depth)
    #
    # imageio.mimwrite(os.path.join("log", args.expname, "test", "img_video.mp4"), images, fps=10, quality=8)
    # imageio.mimwrite(os.path.join("log", args.expname, "test", "depth_video.mp4"), depths, fps=10, quality=8)
def create_render_args(parser):

    parser.add_argument("--device", type=str, default="cuda", help="render device")
    parser.add_argument("--load_pt", type=str, default=None, help="load checkpoint from")
    parser.add_argument("--render_path", default=False, action="store_true")
    parser.add_argument("--interp_0", type=int, default=12)
    parser.add_argument("--interp_1", type=int, default=16)
    parser.add_argument("--num_views", type=int, default=15)
    parser.add_argument("--scan_num", type=int, default=114)
    parser.add_argument("--view_num", type=int, default=20)

    return parser

if __name__ == "__main__":
    parser = config_utils.create_args_parser()
    parser = create_render_args(parser)
    args, unknown = parser.parse_known_args()
    config = config_utils.load_config(args, unknown)
    main(config)

