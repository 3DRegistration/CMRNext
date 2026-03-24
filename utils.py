import argparse
from functools import reduce

import cv2
import mathutils
import torch
import torch.nn.functional as F
import liegroups
import logging
import math
import numpy as np
import scipy
import visibility
from matplotlib import cm
from torch.utils.data.dataloader import default_collate
from torch_scatter import scatter_mean
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid

try:
    import geomstats.geometry.lie_group as lie_group
except ImportError:
    import geomstats.lie_group as lie_group


def init_logger(path, resume=False, save_to_file=False):
    """
    Initialize the logger
    Args:
        path: path to the log file
        resume: if True, the log file is appended to, instead of overwritten
        save_to_file: if True, the log is saved to a file, otherwise only printed on the console
    Returns:
        logger: the logger
    """
    # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # for l in loggers:
    #     l.setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)

    # Set console logging
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    # console_formatter = logging.Formatter(fmt="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # Setup file logging
    if save_to_file:
        mode = 'w'
        if resume:
            mode = 'a'
        file_handler = logging.FileHandler(path, mode=mode)
        file_formatter = logging.Formatter(fmt="%(levelname).1s - %(asctime)s - %(name)s - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('argoverse').setLevel(logging.ERROR)
    return logger


def rotate_points(PC, R, T=None, inverse=True):
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        try:
            RT = T @ R
        except:
            RT = T * R
    else:
        RT = R.copy()
    if inverse:
        RT.invert_safe()
    RT = torch.tensor(RT, device=PC.device, dtype=PC.dtype)

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()

    RT = RT.float()
    PC = PC.float()
    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
        T (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)


def rotate_back(PC_ROTATED, R, T=None):
    """
    Inverse of :func:`~utils.rotate_forward`.
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC_ROTATED, R, T, inverse=False)
    else:
        return rotate_points(PC_ROTATED, R, T, inverse=False)


def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNet to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    try:
        RT = T @ R
    except:
        RT = T * R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT


def merge_inputs(queries):
    point_clouds = []
    imgs = []
    reflectances = []
    car_masks = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'rgb' and key != 'reflectance' and key != 'car_mask'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
        if 'reflectance' in input:
            reflectances.append(input['reflectance'])
        if 'car_mask' in input:
            car_masks.append(input['car_mask'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    if len(reflectances) > 0:
        returns['reflectance'] = reflectances
    if len(car_masks) > 0:
        returns['car_mask'] = car_masks
    return returns


def quaternion_from_matrix(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr+1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()


def euler2mat(z, y, x):
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def euler2quat(roll, pitch, yaw):
    q = np.zeros(4)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr
    return q


def quatmultiply(q, r):
    """
    Multiply two quaternions
    Args:
        q (torch.Tensor/nd.ndarray): shape=[4], first quaternion
        r (torch.Tensor/nd.ndarray): shape=[4], second quaternion

    Returns:
        torch.Tensor: shape=[4], normalized quaternion q*r
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t / t.norm()


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat


def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix

    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin ( rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)


def to_rotation_matrix(R, T):
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT


def overlay_imgs(rgb, lidar, idx=0, pooling=3, max_depth=160., close_thr=25.):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()

    if pooling > 0:
        far_points = lidar.clone()
        far_points[far_points<close_thr] = 0
        close_points = lidar.clone()
        close_points[close_points>=close_thr] = 0.
        close_points[lidar == 0] = 1000.
        close_points = -close_points
        # lidar = F.max_pool2d(lidar, 3, 1, 1)
        close_points = F.max_pool2d(close_points, pooling, 1, 1)
        close_points = -close_points
        close_points[close_points == 1000.] = 0.
        lidar = close_points+far_points

    lidar = lidar[0][0] / max_depth
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    return blended_img


def get_flow(uv, depth, RT, cam_model, image_shape, scale_flow=True, get_valid_indexes=False, al_contrario=True):
    points_3D = torch.zeros((uv.shape[0], 4),
                            device=uv.device, dtype=torch.float)

    # depth to 3D (X-forward)
    points_3D[:, 1] = (uv[:, 0] - cam_model.principal_point[0]) * depth / cam_model.focal_length[0]  # Y
    points_3D[:, 2] = (uv[:, 1] - cam_model.principal_point[1]) * depth / cam_model.focal_length[1]  # Z
    points_3D[:, 0] = depth
    points_3D[:, 3] = 1.

    pc_rotated = rotate_forward(points_3D.clone(), RT)

    uv_rgb = torch.zeros((uv.shape[0], 2), device=uv.device, dtype=torch.float)
    uv_rgb[:, 0] = cam_model.focal_length[0] * pc_rotated[:, 1] / pc_rotated[:, 0] + cam_model.principal_point[0]
    uv_rgb[:, 1] = cam_model.focal_length[1] * pc_rotated[:, 2] / pc_rotated[:, 0] + cam_model.principal_point[1]

    if al_contrario:
        flow = uv - uv_rgb
    else:
        flow = uv_rgb - uv
    flow[uv_rgb < 0] = 0.
    flow[uv_rgb[:, 0] >= image_shape[1]] = 0.
    flow[uv_rgb[:, 1] >= image_shape[0]] = 0.
    
    valid_index = torch.ones_like(flow[:, 0]).bool()
    valid_index = valid_index & (uv_rgb[:, 0] >= 0)
    valid_index = valid_index & (uv_rgb[:, 1] >= 0)
    valid_index = valid_index & (uv_rgb[:, 0] < image_shape[1])
    valid_index = valid_index & (uv_rgb[:, 1] < image_shape[0])
    
    
    # Remove small flow
    flow[abs(flow)<0.5] = 0

    # PWC-Net divides the flow by 20
    if scale_flow:
        flow /= 20.

    # valid_index = ((flow[:, 0] != 0.) & (flow[:, 1] != 0.))
    if get_valid_indexes:
        return flow.clone(), points_3D.clone(), valid_index
    else:
        return flow.clone(), points_3D.clone()


def get_flow_zforward(uv, depth, RT, cam_model, image_shape, scale_flow=True, get_valid_indexes=False, al_contrario=True):
    points_3D = torch.zeros((uv.shape[0], 4),
                            device=uv.device, dtype=torch.float)

    # depth to 3D (X-forward)
    points_3D[:, 0] = (uv[:, 0] - cam_model.principal_point[0]) * depth / cam_model.focal_length[0]  # Y
    points_3D[:, 1] = (uv[:, 1] - cam_model.principal_point[1]) * depth / cam_model.focal_length[1]  # Z
    points_3D[:, 2] = depth
    points_3D[:, 3] = 1.

    pc_rotated = rotate_forward(points_3D.clone(), RT)

    uv_rgb = torch.zeros((uv.shape[0], 2), device=uv.device, dtype=torch.float)
    uv_rgb[:, 0] = cam_model.focal_length[0] * pc_rotated[:, 0] / pc_rotated[:, 2] + cam_model.principal_point[0]
    uv_rgb[:, 1] = cam_model.focal_length[1] * pc_rotated[:, 1] / pc_rotated[:, 2] + cam_model.principal_point[1]

    if al_contrario:
        flow = uv - uv_rgb
    else:
        flow = uv_rgb - uv
    flow[uv_rgb < 0] = 0.
    flow[uv_rgb[:, 0] >= image_shape[1]] = 0.
    flow[uv_rgb[:, 1] >= image_shape[0]] = 0.

    valid_index = torch.ones_like(flow[:, 0]).bool()
    valid_index = valid_index & (uv_rgb[:, 0] >= 0)
    valid_index = valid_index & (uv_rgb[:, 1] >= 0)
    valid_index = valid_index & (uv_rgb[:, 0] < image_shape[1])
    valid_index = valid_index & (uv_rgb[:, 1] < image_shape[0])

    # Remove small flow
    flow[abs(flow) < 0.5] = 0

    # PWC-Net divides the flow by 20
    if scale_flow:
        flow /= 20.

    # valid_index = ((flow[:, 0] != 0.) & (flow[:, 1] != 0.))
    if get_valid_indexes:
        return flow.clone(), points_3D.clone(), valid_index
    else:
        return flow.clone(), points_3D.clone()


def project_points_pose(xyzw, hyp_pose, cam_mat, real_uv):
    """

    :param xyzw: 3D Points tensor, shape 4xN, Z-forward
    :param hyp_pose: Output of solvePnP
    :param cam_mat: intrinsic camera matrix 3x3
    :param real_uv: Real 2D projections tensor, shape 2xN or Nx2
    :return: Projection of 3D points into the camera plane placed in hyp_pose
    """
    if xyzw.shape[1] == 4 and xyzw.shape[0] != 4:
        xyzw = xyzw.t()
    if real_uv.shape[1] == 2:
        real_uv = real_uv.t()
    rot_mat_zf, _ = cv2.Rodrigues(hyp_pose[1])
    RT_pred = torch.zeros((4, 4), dtype=xyzw.dtype)
    RT_pred[:3, :3] = torch.tensor(rot_mat_zf, dtype=xyzw.dtype)
    RT_pred[:3, 3] = torch.tensor(hyp_pose[2], dtype=xyzw.dtype).squeeze()
    RT_pred[3, 3] = 1.
    RT_pred = RT_pred.to(xyzw.device)
    xyzw = torch.mm(RT_pred, xyzw)
    uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
    uv[0, :] = cam_mat[0, 0] * xyzw[0, :] / xyzw[2, :] + cam_mat[0, 2]
    uv[1, :] = cam_mat[1, 1] * xyzw[1, :] / xyzw[2, :] + cam_mat[1, 2]

    diffs = (uv - real_uv).norm(dim=0)

    return diffs


def downsample_flow_and_mask(flow, mask, downsample_ratio, scale_flow=False):
    downsampled_flow = torch.zeros((flow.shape[0]//downsample_ratio, flow.shape[1]//downsample_ratio, 2),
                                   device=flow.device, dtype=torch.float)
    downsampled_flow = visibility.downsample_flow(flow, downsampled_flow, flow.shape[1]//downsample_ratio,
                                           flow.shape[0]//downsample_ratio, downsample_ratio)

    downsampled_mask = torch.zeros((mask.shape[0]//downsample_ratio, mask.shape[1]//downsample_ratio),
                            device=mask.device, dtype=torch.int)
    downsampled_mask = visibility.downsample_mask(mask, downsampled_mask, mask.shape[1]//downsample_ratio,
                                                  mask.shape[0]//downsample_ratio, downsample_ratio)

    if scale_flow:
        downsampled_flow /= downsample_ratio

    return downsampled_flow, downsampled_mask


def downsample_depth(depth, downsample_ratio):
    downsampled_depth = torch.zeros((depth.shape[0]//downsample_ratio, depth.shape[1]//downsample_ratio, 1),
                                   device=depth.device, dtype=torch.float)
    downsampled_depth = visibility.downsample_depth(depth, downsampled_depth, depth.shape[1]//downsample_ratio,
                                           depth.shape[0]//downsample_ratio, downsample_ratio)
    return downsampled_depth


# From HD3
def disp2flow(disp):
    assert disp.size(1) == 1
    padder = torch.zeros(disp.size(), device=disp.device)
    return torch.cat([disp, padder], dim=1)


# From HD3
def resize_dense_vector(vec, des_height, des_width):
    ratio_height = float(des_height / vec.size(2))
    ratio_width = float(des_width / vec.size(3))
    vec = F.interpolate(
        vec, (des_height, des_width), mode='bilinear', align_corners=True)
    if vec.size(1) == 1:
        vec = vec * ratio_width
    else:
        vec = torch.stack(
            [vec[:, 0, :, :] * ratio_width, vec[:, 1, :, :] * ratio_height],
            dim=1)
    return vec


def EndPointError(output, gt):
    # output: [B, 1/2, H, W], stereo or flow prediction
    # gt: [B, C, H, W], 2D ground-truth annotation which may contain a mask
    # NOTE: To benchmark the result, please ensure the ground-truth keeps
    # its ORIGINAL RESOLUTION.
    if output.size(1) == 1:  # stereo
        output = disp2flow(output)
    output = resize_dense_vector(output, gt.size(2), gt.size(3))
    error = torch.norm(output - gt[:, :2, :, :], 2, 1, keepdim=False)
    if gt.size(1) == 3:
        mask = gt[:, 2, :, :]
    else:
        mask = torch.ones_like(error)
    epe = (error * mask).sum() / mask.sum()
    return epe.reshape(1)


def uncertainty_to_color(_tensor):
    """
    Args:
        _tensor: (2, H, W)

    Returns:
        color: (H, W, 3)
    """
    total_uncertainty = _tensor[0, :, :] + _tensor[1, :, :]
    total_uncertainty = torch.from_numpy(total_uncertainty)
    total_uncertainty -= torch.min(total_uncertainty)
    total_uncertainty /= torch.max(total_uncertainty)
    jet = cm.get_cmap('jet')
    color = jet(total_uncertainty)
    color = color[:, :, :3]
    color = F.max_pool2d(torch.tensor(color[:, :, :3]).permute(2, 0, 1).unsqueeze(0), 3, 1, 1)
    color = color[0].permute(1, 2, 0).numpy()
    return color

# From: https://scikit-surgerycore.readthedocs.io
def average_quaternions(quaternions):
    """
    Calculate average quaternion

    :params quaternions: is a Nx4 numpy matrix and contains the quaternions
        to average in the rows.
        The quaternions are arranged as (w,x,y,z), with w being the scalar

    :returns: the average quaternion of the input. Note that the signs
        of the output quaternion can be reversed, since q and -q
        describe the same orientation
    """

    # Number of quaternions to average
    samples = quaternions.shape[0]
    mat_a = np.zeros(shape=(4, 4), dtype=np.float64)

    for i in range(0, samples):
        quat = quaternions[i, :]
        # multiply quat with its transposed version quat' and add mat_a
        mat_a = np.outer(quat, quat) + mat_a

    # scale
    mat_a = (1.0/ samples)*mat_a
    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = np.linalg.eig(mat_a)
    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(np.ravel(eigen_vectors[:, 0]))


def voxelize_gpu(points, voxel_size):
    batch = torch.zeros(points.shape[0], device=points.device, dtype=torch.long)
    voxel_cluster = voxel_grid(points, batch, size = voxel_size)
    voxel_cluster, perm = consecutive_cluster(voxel_cluster)
    batch_sample = batch[perm]
    points_sampled = scatter_mean(points, voxel_cluster, dim = 0)
    
    return points_sampled


# From https://github.com/dilaragokay/central-tendency-rotations/
def hamilton_product(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    return torch.tensor(
        [
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        ]
    )


# From https://github.com/dilaragokay/central-tendency-rotations/
def mean(Q, weights=None):
    if weights is None:
        weights = torch.ones(len(Q), device=torch.device("cuda:0")) / len(Q)
    A = torch.zeros((4, 4), device=torch.device("cuda:0"))
    weight_sum = torch.sum(weights)

    oriented_Q = ((Q[:, 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("bi,bk->bik", (oriented_Q, oriented_Q))
    A = torch.sum(torch.einsum("bij,b->bij", (A, weights)), 0)
    A /= weight_sum

    q_avg = torch.linalg.eigh(A)[1][:, -1]
    if q_avg[0] < 0:
        return -q_avg
    return q_avg

# From https://github.com/dilaragokay/central-tendency-rotations/
def quaternion_median(Q, p=1, max_angular_update=0.0001, max_iterations=1000):
    Q = Q.clone().detach()
    weights = torch.ones(len(Q)) / len(Q)
    q_median = mean(Q, weights)
    Q = Q.cpu()  # because we do sequential operations
    q_median = q_median.cpu()
    EPS_ANGLE = 0.0000001
    max_angular_update = max(max_angular_update, EPS_ANGLE)
    theta = 10 * max_angular_update
    i = 0

    while theta > max_angular_update and i <= max_iterations:
        delta = torch.zeros(3)
        weight_sum = 0
        for q in Q:
            qj = hamilton_product(
                q, torch.tensor([q_median[0], -q_median[1], -q_median[2], -q_median[3]])
            )
            theta = 2 * torch.acos(qj[0])
            if theta > EPS_ANGLE:
                axis_angle = qj[1:] / torch.sin(theta / 2)
                axis_angle *= theta
                weight = 1.0 / pow(theta, 2 - p)
                delta += weight * axis_angle
                weight_sum += weight
        if weight_sum > EPS_ANGLE:
            delta /= weight_sum
            theta = torch.linalg.norm(delta)
            if theta > EPS_ANGLE:
                stby2 = torch.sin(theta * 0.5)
                delta /= theta
                q = torch.tensor(
                    [
                        torch.cos(theta * 0.5),
                        stby2 * delta[0],
                        stby2 * delta[1],
                        stby2 * delta[2],
                    ]
                )
                q_median = hamilton_product(q, q_median)
                if q_median[0] < 0:
                    q_median *= -1
        else:
            theta = 0
        i += 1
    return q_median


# From https://github.com/dilaragokay/central-tendency-rotations/
def quaternion_mode(Q, precision=3):
    if isinstance(Q, torch.Tensor):
        Q = Q.clone().detach()
    else:
        Q = torch.tensor(Q)
    rounded_quats = torch.tensor(np.around(Q.numpy(), precision))
    values, counts = torch.unique(rounded_quats, dim=0, return_counts=True)
    return values[torch.argmax(counts)]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_flow_rgb2lidar(*args):
    pass