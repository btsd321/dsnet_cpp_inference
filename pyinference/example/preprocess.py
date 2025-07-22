# 本文件实现DiffusionSutionNet的预处理功能，主要包括以下步骤：
# 1. 读取输入RGB图像(PNG格式)、深度图像(PNG格式)、Mask图像(PNG格式)、相机信息文件(YAML格式)
# 2. 计算点云及其对应的法向量

import os
import cv2
import numpy as np
import yaml
import json
import torch
import camera_info
# PointNet2操作库，用于点云采样
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
import open3d as o3d  # 3D几何处理

INPUT_TARGET_POINT_NUM = 16384  # 输入点云的目标点数

# 默认路径，可以被函数参数覆盖
default_params_path = '/home/lixinlong/Project/pose_detect_train/example/input/parameter.json'

def _load_parameters(params_file_name):
    """
    加载配置文件
    
    参数:
        params_file_name (str): JSON配置文件路径
        
    返回:
        dict: 包含深度范围等配置的字典
    """
    params = {}
    with open(params_file_name, 'r') as f:
        config = json.load(f)
        params = config
    return params 

def _depth_to_pointcloud_optimized(cam_info, params, us, vs, zs, to_mm=False, xyz_limit=None):
    """
    将深度图像像素坐标转换为3D点云坐标
    
    使用相机内参进行投影变换，从2D像素坐标和深度值重建3D空间坐标。
    这是整个数据处理管道的基础步骤。
    
    参数:
        us (numpy.ndarray): u坐标数组（像素水平坐标）
        vs (numpy.ndarray): v坐标数组（像素垂直坐标）  
        zs (numpy.ndarray): 深度值数组（归一化深度）
        to_mm (bool): 是否转换为毫米单位，默认False（米单位）
        xyz_limit (list): 3D空间裁剪范围，格式[[xmin,xmax], [ymin,ymax], [zmin,zmax]]
                            用于过滤工作空间外的点
    
    返回:
        numpy.ndarray: 3D点云坐标，形状为(N, 3)
    """
    assert len(us) == len(vs) == len(zs), "坐标数组长度必须一致"
    
    # 从参数配置中获取相机内参
    fx = cam_info.intrinsic_matrix[0, 0]
    fy = cam_info.intrinsic_matrix[1, 1]
    cx = cam_info.intrinsic_matrix[0, 2]  # x方向主点坐标
    cy = cam_info.intrinsic_matrix[1, 2]  # y方向主点坐标
    clip_start = params['clip_start']  # 近裁剪面距离
    clip_end = params['clip_end']      # 远裁剪面距离
    
    # 将归一化深度值转换为真实距离（米）
    # 深度图中的值通常是归一化的，需要映射到真实距离范围
    Zline = clip_start + (zs/params['max_val_in_depth']) * (clip_end - clip_start)
    
    # 考虑透视投影的距离校正
    # 校正由于透视投影导致的距离失真
    Zcs = Zline/np.sqrt(1+ np.power((us-cx)/fx,2) + np.power((vs-cy)/fy,2))
    
    # 可选：转换为毫米单位（某些应用需要）
    if to_mm:
        Zcs *= 1000
        
    # 使用针孔相机模型进行3D重建
    # X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
    Xcs = (us - cx) * Zcs / fx
    Ycs = (vs - cy) * Zcs / fy
    
    # 重塑为列向量并组合成点云
    Xcs = np.reshape(Xcs, (-1, 1))
    Ycs = np.reshape(Ycs, (-1, 1))
    Zcs = np.reshape(Zcs, (-1, 1))
    points = np.concatenate([Xcs, Ycs, Zcs], axis=-1)
    
    # 可选：根据xyz范围裁剪点云，去除工作空间外的无关点
    if xyz_limit is not None:
        # X轴裁剪
        if xyz_limit[0] is not None:
            xmin, xmax = xyz_limit[0]
            if xmin is not None:
                idx = np.where(points[:, 0] > xmin)
                points = points[idx]
            if xmax is not None:
                idx = np.where(points[:, 0] < xmax)
                points = points[idx]
        # Y轴裁剪
        if xyz_limit[1] is not None:
            ymin, ymax = xyz_limit[1]
            if ymin is not None:
                idx = np.where(points[:, 1] > ymin)
                points = points[idx]
            if ymax is not None:
                idx = np.where(points[:, 1] < ymax)
                points = points[idx]
        # Z轴裁剪
        if xyz_limit[2] is not None:
            zmin, zmax = xyz_limit[2]
            if zmin is not None:
                idx = np.where(points[:, 2] > zmin)
                points = points[idx]
            if zmax is not None:
                idx = np.where(points[:, 2] < zmax)
                points = points[idx]
                
    return points

def preprocess(input_rgb_path, input_depth_path, input_mask_path, camera_info_path, params_path=None):
    if params_path is None:
        params_path = default_params_path
    # 读取RGB图像
    rgb_img = cv2.imread(input_rgb_path)
    # 读取深度图像
    depth_img = cv2.imread(input_depth_path, cv2.IMREAD_ANYDEPTH)
    # 读取Mask图像，白色为物体，其他为背景
    mask_img = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
    valid_mask = mask_img == 255  # 白色区域为物体
    
    # 读取相机信息
    cam_info = camera_info.get_camera_info_from_yaml(camera_info_path)
    
    # 读取参数配置
    params = _load_parameters(params_path)
    
    # 生成点云及其法向量
    xs, ys = np.where(valid_mask)
    zs = depth_img[valid_mask]
    
    # 执行3D重建：像素坐标 + 深度 → 3D点云
    points = _depth_to_pointcloud_optimized(cam_info, params, xs, ys, zs, to_mm=False, xyz_limit=None)
    
    # === 第4步：点云采样和标准化 ===
    num_pnt = points.shape[0]
    if num_pnt == 0:
        raise ValueError('没有前景点，跳过当前场景！')
        
    # 情况1：点数不足，通过重复采样达到目标数量
    if num_pnt <= INPUT_TARGET_POINT_NUM:
        t = int(1.0 * INPUT_TARGET_POINT_NUM / num_pnt) + 1
        points_tile = np.tile(points, [t, 1])  # 重复点云
        points = points_tile[:INPUT_TARGET_POINT_NUM]       
    # 情况2：点数过多，使用最远点采样(FPS)进行下采样
    elif num_pnt > INPUT_TARGET_POINT_NUM:
        # 转换为PyTorch张量并移到GPU（如果可用）
        points_transpose = torch.from_numpy(points.reshape(1, points.shape[0], points.shape[1])).float()
        if torch.cuda.is_available():
            points_transpose = points_transpose.cuda()
        
        # 执行最远点采样，保持点云的几何分布
        sampled_idx = furthest_point_sample(points_transpose, INPUT_TARGET_POINT_NUM).cpu().numpy().reshape(INPUT_TARGET_POINT_NUM)
        points = points[sampled_idx]
    else:
        pass
    
    # 构建Open3D点云对象用于法向量计算
    pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    
    # 使用半径搜索估计每个点的表面法向量
    # 半径0.015米是经验值，平衡计算精度和效率
    pc_o3d.estimate_normals(
        o3d.geometry.KDTreeSearchParamRadius(0.015), 
        fast_normal_computation=False  # 使用精确计算保证质量
    )
    
    # 统一法向量方向：都指向负Z轴方向（向下）
    # 这对吸取任务很重要，因为吸盘通常从上往下接近物体
    pc_o3d.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    pc_o3d.normalize_normals()  # 标准化为单位向量
    
    # 提取处理后的数据
    suction_points = points  # 吸取候选点
    suction_or = np.array(pc_o3d.normals).astype(np.float32)  # 对应的法向量
    
    return suction_points, suction_or
    
    

