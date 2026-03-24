import argparse
import os
import math
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import open3d as o3d

from camera_model import CameraModel
from evaluate_flow_calibration import prepare_input
from models.get_model import get_model
from utils import (quat2mat, tvector2mat, quaternion_from_matrix, rotate_forward,
                   get_flow_zforward, downsample_depth, EndPointError, voxelize_gpu)

# NOTE: This demo is a simplified adaptation of evaluate_flow_localization.py.
# It performs iterative pose refinement given: an RGB image, a LiDAR point cloud and camera intrinsics.
# It outputs the estimated pose correction w.r.t. a provided initial extrinsic (usually LiDAR->Camera).
# The checkpoint(s) should be trained CMRNext weights.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    points = None
    if ext == '.pcd':
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
    elif ext == '.bin':  # KITTI style
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        points = pts[:, :3]
    elif ext == '.npy':
        arr = np.load(path)
        if arr.shape[1] >= 3:
            points = arr[:, :3].astype(np.float32)
    elif ext == '.h5':
        import h5py
        with h5py.File(path, 'r') as hf:
            if 'PC' in hf:
                points = hf['PC'][:].astype(np.float32)
            else:
                # Assume dataset only contains xyz
                first_key = list(hf.keys())[0]
                points = hf[first_key][:, :3].astype(np.float32)
    else:
        raise RuntimeError(f'Unsupported point cloud format: {ext}')

    if points is None or points.shape[0] == 0:
        raise RuntimeError('Empty point cloud')

    pc = torch.from_numpy(points.T)  # 3 x N
    ones = torch.ones(1, pc.shape[1])
    pc = torch.cat([pc, ones], dim=0)  # 4 x N homogeneous
    return pc


def parse_calibration(calib_path, fx=None, fy=None, cx=None, cy=None):
    if calib_path is not None:
        with open(calib_path, 'r') as f:
            data = yaml.safe_load(f)
        fx = data.get('fx', fx)
        fy = data.get('fy', fy)
        cx = data.get('cx', cx)
        cy = data.get('cy', cy)
        initial_extrinsic = data.get('initial_extrinsic', None)
    else:
        initial_extrinsic = None
    if None in [fx, fy, cx, cy]:
        raise RuntimeError('Camera intrinsics not fully specified (fx, fy, cx, cy).')
    K = torch.tensor([fx, fy, cx, cy]).float()
    extrinsic = None
    if initial_extrinsic is not None:
        extrinsic = torch.tensor(initial_extrinsic, dtype=torch.float32).view(4, 4)
    return K, extrinsic


def quaternion_distance_deg(q, r):
    # Both torch tensors (4,)
    dot = torch.abs(torch.dot(q / q.norm(), r / r.norm())).clamp(max=1.0)
    return 2 * torch.arccos(dot) * 180.0 / math.pi


def build_config_from_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    # Fill defaults used by evaluation scripts
    defaults = dict(uncertainty=False, fourier_levels=-1, num_scales=1, der_type='NLL', unc_freeze=False,
                    voxelize=0.1, context_encoder='rgb', al_contrario=cfg.get('al_contrario', True))
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    return ckpt, cfg


def prepare_rgb(image_path, normalize, mean_torch, std_torch):
    rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f'Cannot read image {image_path}')
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_t = torch.from_numpy(rgb).float() / 255.0
    rgb_t = rgb_t.permute(2, 0, 1)  # C,H,W
    rgb_t = rgb_t.to(mean_torch.device)  # Ensure same device as mean/std
    if normalize:
        rgb_t = (rgb_t - mean_torch) / std_torch
    return rgb_t


def pad_to_shape(tensor, target_shape):
    # tensor: C,H,W -> pad bottom and right
    _, H, W = tensor.shape
    th, tw = target_shape
    pad_h = th - H
    pad_w = tw - W
    if pad_h < 0 or pad_w < 0:
        raise RuntimeError('Target shape smaller than tensor shape')
    return F.pad(tensor, [0, pad_w, 0, pad_h])


def run_demo(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load checkpoint(s)
    weight_paths = args.weights
    ckpt, ckpt_cfg = build_config_from_checkpoint(weight_paths[0])

    # Build unified config
    _config = {}
    for key in ['use_reflectance', 'initial_pool', 'upsample_method', 'occlusion_kernel', 'occlusion_threshold',
                'al_contrario', 'amp', 'scaled_gt', 'max_depth', 'normalize_images', 'uncertainty', 'fourier_levels',
                'num_scales', 'der_type', 'unc_freeze', 'voxelize', 'context_encoder']:
        _config[key] = ckpt_cfg.get(key, _config.get(key, None))
    _config['al_contrario'] = ckpt_cfg.get('al_contrario', True)
    _config['use_reflectance'] = ckpt_cfg.get('use_reflectance', False)

    mean_torch = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_torch = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Camera intrinsics & initial extrinsic
    calib, initial_extrinsic = parse_calibration(args.calib, args.fx, args.fy, args.cx, args.cy)
    calib = calib.to(device)

    # Load image & point cloud
    pc = load_point_cloud(args.pointcloud).to(device)
    if _config['max_depth'] < 100.:
        pc = pc[:, pc[0, :] < _config['max_depth']]
    if args.voxel_size > 0:
        pc = voxelize_gpu(pc[:3, :].T, args.voxel_size).T
        ones = torch.ones(1, pc.shape[1], device=pc.device)
        pc = torch.cat([pc, ones], dim=0)

    rgb = prepare_rgb(args.image, _config['normalize_images'], mean_torch.to(device), std_torch.to(device)).to(device)
    real_shape = (rgb.shape[1], rgb.shape[2])  # H,W

    # Determine network input shape (multiple of 64)
    net_H = (real_shape[0] + 63) // 64 * 64
    net_W = (real_shape[1] + 63) // 64 * 64
    img_shape = (net_H, net_W)

    # Always start from identity (no artificial misalignment)
    rot_err = torch.tensor([1., 0., 0., 0.], device=device)
    tr_err = torch.zeros(3, device=device)

    R_err = quat2mat(rot_err)
    T_err = tvector2mat(tr_err)
    RT_err = torch.mm(T_err, R_err)  # Pose error to estimate (initial)

    # Rotate point cloud to create misalignment
    pc_rotated = rotate_forward(pc, RT_err.inverse())  # identity -> unchanged

    # Project and prepare first inputs
    cam_params = calib
    reflectance = None  # Not supported in simple demo
    depth_img_no_occ, uv, indexes, depth = prepare_input(cam_params, pc_rotated, (real_shape[0], real_shape[1], 3),
                                                         reflectance, dict(max_depth=_config['max_depth'],
                                                                           use_reflectance=_config['use_reflectance'],
                                                                           occlusion_threshold=_config['occlusion_threshold'],
                                                                           occlusion_kernel=_config['occlusion_kernel']),
                                                         change_frame=False)

    # Build camera model for flow computation
    cam_model = CameraModel()
    cam_model.focal_length = calib[:2]
    cam_model.principal_point = calib[2:]

    flow, points_3D, valid_idx = get_flow_zforward(uv.float(), depth[indexes], RT_err, cam_model,
                                                   [real_shape[0], real_shape[1], 3],
                                                   scale_flow=False, al_contrario=_config['al_contrario'],
                                                   get_valid_indexes=True)
    uv = uv[valid_idx]
    flow = flow[valid_idx]
    points_3D = points_3D[valid_idx]

    # Visualization: Projected point cloud on input image
    img_input = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_proj = img_input.copy()
    for pt in uv.cpu().numpy():
        cv2.circle(img_proj, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.title('Projected Point Cloud on Input Image')
    plt.imshow(img_proj)
    plt.axis('off')
    plt.savefig('projected_pointcloud_on_input.png')
    plt.close()

    # Prepare network tensors (pad & downsample like eval script)
    flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device=device)
    flow_img[uv[:, 1], uv[:, 0]] = flow
    flow_mask = torch.zeros((real_shape[0], real_shape[1]), device=device, dtype=torch.int)
    flow_mask[uv[:, 1], uv[:, 0]] = 1

    rgb_in = pad_to_shape(rgb, img_shape)
    depth_in = pad_to_shape(depth_img_no_occ, img_shape)
    flow_img = pad_to_shape(flow_img.permute(2, 0, 1), img_shape).permute(1, 2, 0)
    flow_mask = pad_to_shape(flow_mask.unsqueeze(0), img_shape)[0]

    # Network expects batches
    lidar_input = depth_in.unsqueeze(0).repeat(1, 25, 1, 1)  # [1, 25, H, W]
    rgb_input = rgb_in.unsqueeze(0)

    # Load model(s)
    models = []
    for wp in weight_paths:
        ckpt_w, cfg_w = build_config_from_checkpoint(wp)
        model = get_model(cfg_w, img_shape)
        model.load_state_dict(ckpt_w['state_dict'], strict=False)
        model.to(device).eval()
        models.append(model)

    current_RT = RT_err.clone()  # Ground-truth unknown; refine from this guess

    for iteration, model in enumerate(models, start=1):
        with torch.no_grad():
            pred_flow_tuple = model(rgb_input, lidar_input, iters=24)
            pred_flow_list, _ = pred_flow_tuple
            up_flow = pred_flow_list[-1][0]  # (2,H,W)
        # Apply flow to uv
        up_flow_hw = up_flow.permute(1, 2, 0)
        new_uv = uv.float()
        if _config['al_contrario']:
            new_uv = new_uv - up_flow_hw[uv[:, 1], uv[:, 0]]
        else:
            new_uv = new_uv + up_flow_hw[uv[:, 1], uv[:, 0]]

        # PnP with OpenCV (CPU)
        points_2d = new_uv.detach().cpu().numpy().astype(np.float32)
        obj_points = points_3D[:, :3].detach().cpu().numpy().astype(np.float32)
        Kmat = np.array([[calib[0].item(), 0, calib[2].item()],
                         [0, calib[1].item(), calib[3].item()],
                         [0, 0, 1]], dtype=np.float32)
        dist = np.zeros(4)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, points_2d, Kmat, dist,
                                                           iterationsCount=200, reprojectionError=2.0, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            print('PnP failed at iteration', iteration)
            break
        Rmat, _ = cv2.Rodrigues(rvec)
        rot_mat = torch.from_numpy(Rmat).float()
        transl = torch.from_numpy(tvec.squeeze()).float()
        pred_quat = quaternion_from_matrix(rot_mat)
        R_pred = quat2mat(pred_quat).to(device)
        T_pred = tvector2mat(transl.to(device))
        RT_pred = torch.mm(T_pred, R_pred)
        # Compose update (similar logic to eval script)
        current_RT = torch.mm(current_RT, RT_pred.inverse())
        # Update rotated point cloud & re-build inputs for next iteration (if any)
        if iteration < len(models):
            pc_rotated = rotate_forward(pc, current_RT.inverse())
            depth_img_no_occ, uv, indexes, depth = prepare_input(cam_params, pc_rotated, (real_shape[0], real_shape[1], 3),
                                                                 reflectance, dict(max_depth=_config['max_depth'],
                                                                                   use_reflectance=_config['use_reflectance'],
                                                                                   occlusion_threshold=_config['occlusion_threshold'],
                                                                                   occlusion_kernel=_config['occlusion_kernel']),
                                                                 change_frame=False)
            flow, points_3D, valid_idx = get_flow_zforward(uv.float(), depth[indexes], current_RT, cam_model,
                                                           [real_shape[0], real_shape[1], 3],
                                                           scale_flow=False, al_contrario=_config['al_contrario'],
                                                           get_valid_indexes=True)
            uv = uv[valid_idx]
            flow = flow[valid_idx]
            points_3D = points_3D[valid_idx]
            flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device=device)
            flow_img[uv[:, 1], uv[:, 0]] = flow
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device=device, dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = 1
            depth_in = pad_to_shape(depth_img_no_occ, img_shape)
            rgb_input = pad_to_shape(rgb, img_shape).unsqueeze(0)
            lidar_input = depth_in.unsqueeze(0)

        # Report per-iteration
        tr_est = current_RT[:3, 3]
        rot_est_quat = quaternion_from_matrix(current_RT)
    print(f'Iteration {iteration}: Translation (m) = {tr_est.cpu().numpy()}, Quaternion = {rot_est_quat.cpu().numpy()}')

    # Final pose correction relative to initial (if provided)
    print('\nEstimated relative pose (map -> camera) matrix current_RT^{-1}:')
    print(current_RT.inverse().cpu().numpy())
    print('\nInternal tracking matrix current_RT (camera -> map correction chain):')
    print(current_RT.cpu().numpy())
    if initial_extrinsic is not None:
        refined_extrinsic = torch.mm(current_RT.inverse(), initial_extrinsic.to(device))
        print('\nIf initial_extrinsic given (map->camera guess), refined_extrinsic = current_RT^{-1} * initial_extrinsic:')
        print(refined_extrinsic.cpu().numpy())


def build_argparser():
    p = argparse.ArgumentParser(description='CMRNext Single Image/PointCloud Localization Demo (no artificial misalignment)')
    p.add_argument('--image', required=True, help='Path to RGB image file')
    p.add_argument('--pointcloud', required=True, help='Path to point cloud file (.pcd/.bin/.npy/.h5)')
    p.add_argument('--calib', required=False, default=None, help='Calibration YAML with fx,fy,cx,cy,initial_extrinsic (optional)')
    p.add_argument('--fx', type=float, default=None)
    p.add_argument('--fy', type=float, default=None)
    p.add_argument('--cx', type=float, default=None)
    p.add_argument('--cy', type=float, default=None)
    p.add_argument('--weights', nargs='+', required=True, help='Model checkpoint(s) for iterative refinement')
    p.add_argument('--voxel_size', type=float, default=0.0, help='Voxel size (meters) for optional point cloud downsampling')
    p.add_argument('--seed', type=int, default=42)
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    run_demo(args)
