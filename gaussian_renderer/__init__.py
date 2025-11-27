#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

def render_fastgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult, scaling_modifier = 1.0, override_color = None, get_flag=None, metric_map = None):
    """
    FastGS渲染函数：将3D高斯场景渲染为2D图像
    
    功能描述：
        使用可微光栅化技术将3D高斯点云渲染成2D图像。该函数是FastGS的核心渲染函数，
        支持球谐函数颜色表示、协方差矩阵计算、视锥体剔除等功能。
    
    参数说明：
        viewpoint_camera: 视点相机对象，包含相机的内外参数、视图矩阵、投影矩阵等
        pc (GaussianModel): 高斯模型对象，包含所有高斯点的位置、颜色、不透明度、缩放、旋转等属性
        pipe: 渲染管线配置对象，包含是否在Python中计算协方差、是否预计算颜色等选项
        bg_color (torch.Tensor): 背景颜色张量，形状: (3,)，必须在GPU上
        mult: FastGS的缩放乘数参数，用于控制高斯的缩放
        scaling_modifier (float): 缩放修正因子，默认为1.0，用于临时调整所有高斯的大小
        override_color (torch.Tensor, optional): 覆盖颜色，形状: (N, 3)，如果提供则使用该颜色而不是球谐函数
        get_flag: 获取标志，用于调试或特殊渲染模式
        metric_map (torch.Tensor, optional): 度量图，形状: (H*W,)，用于记录每个像素的统计信息
    
    返回值：
        dict: 包含以下键值对的字典：
            - "render": 渲染的RGB图像，形状: (3, H, W)
            - "viewspace_points": 屏幕空间点坐标（用于梯度计算），形状: (N, 4)
            - "visibility_filter": 可见高斯点的索引，形状: (M, 1)，M为可见点数量
            - "radii": 每个高斯在屏幕空间的半径（像素单位），形状: (N,)
            - "accum_metric_counts": 累积的度量计数，用于统计每个高斯的渲染贡献
    
    注意：
        背景颜色张量(bg_color)必须在GPU上！
    """
 
    # ============ 准备屏幕空间点张量 ============
    # 创建零张量用于存储屏幕空间的2D投影点，该张量需要梯度以便反向传播
    # 形状: (N, 4)，其中N是高斯点数量，4维是为了兼容齐次坐标
    # 这个张量在前向传播中会被光栅化器填充，在反向传播中会收集梯度
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()  # 保留中间变量的梯度（通常PyTorch不保存非叶节点的梯度）
    except:
        pass  # 如果retain_grad失败，忽略异常继续执行

    # ============ 配置光栅化参数 ============
    # 计算视场角的一半的正切值（用于透视投影）
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 水平方向视场角
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 垂直方向视场角

    # 如果没有提供度量图，则创建一个零张量
    # 形状: (H*W,)，每个像素对应一个整数计数
    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')

    # 创建光栅化设置对象，包含所有渲染所需的参数
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 输出图像高度（像素）
        image_width=int(viewpoint_camera.image_width),    # 输出图像宽度（像素）
        tanfovx=tanfovx,  # 水平视场角的tan值
        tanfovy=tanfovy,  # 垂直视场角的tan值
        bg=bg_color,  # 背景颜色，形状: (3,)
        scale_modifier=scaling_modifier,  # 全局缩放修正因子
        viewmatrix=viewpoint_camera.world_view_transform,  # 世界到视图的变换矩阵，形状: (4, 4)
        projmatrix=viewpoint_camera.full_proj_transform,   # 完整的投影矩阵（视图+投影），形状: (4, 4)
        sh_degree=pc.active_sh_degree,  # 当前激活的球谐函数阶数
        campos=viewpoint_camera.camera_center,  # 相机中心在世界坐标系中的位置，形状: (3,)
        mult = mult,  # FastGS的缩放乘数
        prefiltered=False,  # 是否使用预过滤（通常为False）
        debug=pipe.debug,  # 是否启用调试模式
        get_flag=get_flag,  # 获取标志
        metric_map = metric_map  # 度量图
    )

    # 创建光栅化器对象
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ============ 准备高斯点的基本属性 ============
    means3D = pc.get_xyz  # 3D空间中的高斯中心位置，形状: (N, 3)
    means2D = screenspace_points  # 2D屏幕空间中的投影点（待填充），形状: (N, 4)
    opacity = pc.get_opacity  # 不透明度，形状: (N, 1)

    # ============ 处理协方差矩阵 ============
    # 协方差矩阵定义了高斯的形状（大小和朝向）
    # 可以选择在Python中预计算3D协方差矩阵，或者提供缩放和旋转参数让光栅化器计算
    scales = None      # 缩放参数，形状: (N, 3)，表示xyz三个轴的缩放
    rotations = None   # 旋转参数（四元数），形状: (N, 4)
    cov3D_precomp = None  # 预计算的3D协方差矩阵，形状: (N, 6)（上三角矩阵展开）

    if pipe.compute_cov3D_python:
        # 如果配置为在Python中计算协方差矩阵，则预计算（通常用于调试）
        cov3D_precomp = pc.get_covariance(scaling_modifier)  # 形状: (N, 6)
    else:
        # 否则提供缩放和旋转参数，让CUDA光栅化器高效计算（推荐方式）
        scales = pc.get_scaling  # 形状: (N, 3)
        rotations = pc.get_rotation  # 形状: (N, 4)，四元数表示

    # ============ 处理颜色信息 ============
    # 高斯的颜色可以通过球谐函数(SH)表示，也可以直接提供RGB颜色
    # 球谐函数可以表示视角相关的颜色（类似于表面的反射特性）
    shs = None  # 球谐系数（除了DC分量），形状: (N, (max_sh_degree+1)^2 - 1, 3)
    colors_precomp = None  # 预计算的RGB颜色，形状: (N, 3)
    
    if override_color is None:
        # 如果没有覆盖颜色，则使用球谐函数或其计算结果
        if pipe.convert_SHs_python:
            # 选项1：在Python中将球谐函数转换为RGB（通常用于调试）
            # 重排球谐系数的维度：(N, num_features, 3) -> (N, 3, num_features)
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            
            # 计算从高斯点到相机的方向向量，形状: (N, 3)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            
            # 归一化方向向量，形状: (N, 3)
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            # 使用球谐函数计算该方向的RGB颜色，形状: (N, 3)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            
            # 将颜色范围调整到[0, +∞)（球谐函数输出可能为负）
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # 形状: (N, 3)
        else:
            # 选项2：提供球谐系数给光栅化器，让CUDA内核进行转换（推荐方式，更高效）
            dc, shs = pc.get_features_dc, pc.get_features_rest
            # dc: 球谐的DC分量（第0阶），形状: (N, 1, 3)
            # shs: 球谐的其他分量（第1阶及以上），形状: (N, (max_sh_degree+1)^2 - 1, 3)
    else:
        # 如果提供了覆盖颜色，则直接使用该颜色（忽略球谐函数）
        colors_precomp = override_color  # 形状: (N, 3)

    # ============ 执行光栅化 ============
    # 将3D高斯点云光栅化为2D图像
    # 光栅化过程：投影 -> 视锥体剔除 -> 排序 -> α混合
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D = means3D,        # 3D中心位置，形状: (N, 3)
        means2D = means2D,        # 2D屏幕空间点（用于梯度），形状: (N, 4)
        dc = dc,                  # 球谐DC分量，形状: (N, 1, 3)（如果不使用则为None）
        shs = shs,                # 球谐其他分量，形状: (N, K, 3)（如果不使用则为None）
        colors_precomp = colors_precomp,  # 预计算颜色，形状: (N, 3)（如果使用SH则为None）
        opacities = opacity,      # 不透明度，形状: (N, 1)
        scales = scales,          # 缩放参数，形状: (N, 3)（如果使用预计算协方差则为None）
        rotations = rotations,    # 旋转参数，形状: (N, 4)（如果使用预计算协方差则为None）
        cov3D_precomp = cov3D_precomp)  # 预计算的3D协方差，形状: (N, 6)（如果提供了scales和rotations则为None）
    
    # 返回值：
    # - rendered_image: 渲染的RGB图像，形状: (3, H, W)
    # - radii: 每个高斯在屏幕上的半径（像素单位），形状: (N,)，0表示该高斯不可见
    # - accum_metric_counts: 累积度量计数，用于FastGS的统计

    # ============ 返回渲染结果 ============
    # 被视锥体剔除或半径为0的高斯点是不可见的
    # 它们会被排除在用于密集化分裂标准的梯度更新之外
    return {"render": rendered_image,  # 渲染的RGB图像，形状: (3, H, W)
            "viewspace_points": screenspace_points,  # 屏幕空间点（带梯度），形状: (N, 4)
            "visibility_filter" : (radii > 0).nonzero(),  # 可见高斯的索引，形状: (M, 1)
            "radii": radii,  # 屏幕空间半径，形状: (N,)
            "accum_metric_counts" : accum_metric_counts}  # 累积度量计数