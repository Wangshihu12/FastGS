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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    """
    可微高斯光栅化自定义autograd函数
    
    功能描述：
        实现3D高斯溅射的可微光栅化操作，支持自动求导。
        该类继承自torch.autograd.Function，定义了前向传播和反向传播的CUDA实现。
        前向传播将3D高斯投影到2D图像，反向传播计算所有可学习参数的梯度。
    """
    
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings
    ):
        """
        前向传播：将3D高斯渲染为2D图像
        
        参数说明：
            ctx: PyTorch上下文对象，用于保存反向传播所需的张量
            means3D (torch.Tensor): 3D空间中高斯的中心位置，形状: (N, 3)
            means2D (torch.Tensor): 2D屏幕空间中的投影点（用于梯度计算），形状: (N, 4)
            dc (torch.Tensor): 球谐函数的DC分量（第0阶），形状: (N, 1, 3)
            sh (torch.Tensor): 球谐函数的其他分量（第1阶及以上），形状: (N, K, 3)，K取决于阶数
            colors_precomp (torch.Tensor): 预计算的RGB颜色（如果不使用球谐），形状: (N, 3)
            opacities (torch.Tensor): 不透明度，形状: (N, 1)
            scales (torch.Tensor): 缩放参数（xyz三轴），形状: (N, 3)
            rotations (torch.Tensor): 旋转参数（四元数），形状: (N, 4)
            cov3Ds_precomp (torch.Tensor): 预计算的3D协方差矩阵（上三角展开），形状: (N, 6)
            raster_settings: 光栅化设置对象，包含相机参数、图像尺寸等配置
        
        返回值：
            tuple: 包含三个元素的元组
                - color (torch.Tensor): 渲染的RGB图像，形状: (3, H, W)
                - radii (torch.Tensor): 每个高斯在屏幕空间的半径（像素单位），形状: (N,)
                - accum_metric_counts: 累积度量计数，用于FastGS的统计分析
        """

        # ============ 准备CUDA函数参数 ============
        # 重组参数以符合C++/CUDA库的预期格式
        get_flag = raster_settings.get_flag
        if get_flag == None:
            get_flag = False  # 默认不获取额外标志

        # 按照CUDA内核期望的顺序排列参数
        args = (
            raster_settings.bg,          # 背景颜色，形状: (3,)
            means3D,                     # 3D中心位置，形状: (N, 3)
            colors_precomp,              # 预计算颜色，形状: (N, 3)或空
            opacities,                   # 不透明度，形状: (N, 1)
            scales,                      # 缩放，形状: (N, 3)或空
            rotations,                   # 旋转，形状: (N, 4)或空
            raster_settings.scale_modifier,  # 全局缩放修正因子
            cov3Ds_precomp,              # 预计算协方差，形状: (N, 6)或空
            raster_settings.metric_map,  # 度量图，形状: (H*W,)
            raster_settings.viewmatrix,  # 视图矩阵，形状: (4, 4)
            raster_settings.projmatrix,  # 投影矩阵，形状: (4, 4)
            raster_settings.tanfovx,     # 水平视场角的tan值
            raster_settings.tanfovy,     # 垂直视场角的tan值
            raster_settings.image_height,  # 图像高度（像素）
            raster_settings.image_width,   # 图像宽度（像素）
            dc,                          # 球谐DC分量，形状: (N, 1, 3)或空
            sh,                          # 球谐其他分量，形状: (N, K, 3)或空
            raster_settings.sh_degree,   # 球谐阶数
            raster_settings.campos,      # 相机位置，形状: (3,)
            raster_settings.mult,        # FastGS缩放乘数
            raster_settings.prefiltered, # 是否预过滤
            raster_settings.debug,       # 是否调试模式
            get_flag                     # 获取标志
        )

        # ============ 调用CUDA光栅化器 ============
        if raster_settings.debug:
            # 调试模式：在调用CUDA前复制参数到CPU，以便错误时保存快照
            cpu_args = cpu_deep_copy_tuple(args)  # 深拷贝到CPU（防止GPU内存被破坏）
            try:
                # 调用C++/CUDA扩展的光栅化函数
                num_rendered, num_buckets, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                # 如果发生错误，保存参数快照用于调试
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            # 正常模式：直接调用CUDA光栅化函数
            # 返回值：
            # - num_rendered: 实际渲染的高斯数量（可见的高斯）
            # - num_buckets: 用于并行渲染的分桶数量
            # - color: 渲染的RGB图像，形状: (3, H, W)
            # - radii: 每个高斯的屏幕空间半径，形状: (N,)
            # - geomBuffer: 几何缓冲区（存储投影后的几何信息，用于反向传播）
            # - binningBuffer: 分桶缓冲区（存储排序信息，用于反向传播）
            # - imgBuffer: 图像缓冲区（存储渲染中间结果，用于反向传播）
            # - sampleBuffer: 采样缓冲区（FastGS特有，用于记录采样信息）
            # - accum_metric_counts: 累积度量计数
            num_rendered, num_buckets, color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, accum_metric_counts = _C.rasterize_gaussians(*args)

        # ============ 保存反向传播所需的张量 ============
        # 将光栅化设置和中间结果保存到上下文中，供backward使用
        ctx.raster_settings = raster_settings  # 保存渲染配置
        ctx.num_rendered = num_rendered  # 保存可见高斯数量
        ctx.num_buckets = num_buckets    # 保存分桶数量
        
        # 保存需要计算梯度的张量和中间缓冲区
        # 这些张量将在反向传播中用于计算梯度
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer)
        
        # 返回渲染结果
        return color, radii, accum_metric_counts

    @staticmethod
    def backward(ctx, grad_out_color, _, g_metric):
        """
        反向传播：计算所有可学习参数的梯度
        
        功能描述：
            根据输出图像的梯度，反向计算3D高斯所有参数（位置、颜色、不透明度等）的梯度。
            使用前向传播时保存的中间缓冲区，调用CUDA实现的反向传播函数。
        
        参数说明：
            ctx: PyTorch上下文对象，包含前向传播保存的张量
            grad_out_color (torch.Tensor): 渲染图像的梯度（来自损失函数），形状: (3, H, W)
            _: radii的梯度（不需要，占位符）
            g_metric: accum_metric_counts的梯度（不需要，占位符）
        
        返回值：
            tuple: 包含所有输入参数的梯度，顺序与forward的输入参数一致
                - grad_means3D: 3D位置的梯度，形状: (N, 3)
                - grad_means2D: 2D投影点的梯度，形状: (N, 4)
                - grad_dc: 球谐DC分量的梯度，形状: (N, 1, 3)
                - grad_sh: 球谐其他分量的梯度，形状: (N, K, 3)
                - grad_colors_precomp: 预计算颜色的梯度，形状: (N, 3)
                - grad_opacities: 不透明度的梯度，形状: (N, 1)
                - grad_scales: 缩放参数的梯度，形状: (N, 3)
                - grad_rotations: 旋转参数的梯度，形状: (N, 4)
                - grad_cov3Ds_precomp: 协方差矩阵的梯度，形状: (N, 6)
                - None: raster_settings的梯度（配置对象不需要梯度）
        """

        # ============ 从上下文恢复前向传播的信息 ============
        num_rendered = ctx.num_rendered  # 可见高斯数量
        num_buckets = ctx.num_buckets    # 分桶数量
        raster_settings = ctx.raster_settings  # 光栅化设置
        
        # 恢复前向传播时保存的张量和缓冲区
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = ctx.saved_tensors

        # ============ 准备CUDA反向传播函数的参数 ============
        # 按照C++反向传播函数期望的顺序重组参数
        args = (raster_settings.bg,           # 背景颜色，形状: (3,)
                means3D,                      # 3D位置，形状: (N, 3)
                radii,                        # 屏幕空间半径，形状: (N,)
                colors_precomp,               # 预计算颜色，形状: (N, 3)或空
                scales,                       # 缩放，形状: (N, 3)或空
                rotations,                    # 旋转，形状: (N, 4)或空
                raster_settings.scale_modifier,  # 全局缩放修正因子
                cov3Ds_precomp,               # 预计算协方差，形状: (N, 6)或空
                raster_settings.viewmatrix,   # 视图矩阵，形状: (4, 4)
                raster_settings.projmatrix,   # 投影矩阵，形状: (4, 4)
                raster_settings.tanfovx,      # 水平视场角的tan值
                raster_settings.tanfovy,      # 垂直视场角的tan值
                grad_out_color,               # 输出图像的梯度，形状: (3, H, W)
                dc,                           # 球谐DC分量，形状: (N, 1, 3)或空
                sh,                           # 球谐其他分量，形状: (N, K, 3)或空
                raster_settings.sh_degree,    # 球谐阶数
                raster_settings.campos,       # 相机位置，形状: (3,)
                geomBuffer,                   # 几何缓冲区（前向传播保存的中间结果）
                num_rendered,                 # 可见高斯数量
                binningBuffer,                # 分桶缓冲区（前向传播保存的排序信息）
                imgBuffer,                    # 图像缓冲区（前向传播保存的渲染中间结果）
                num_buckets,                  # 分桶数量
                sampleBuffer,                 # 采样缓冲区（FastGS特有）
                raster_settings.debug)        # 是否调试模式

        # ============ 调用CUDA反向传播函数计算梯度 ============
        if raster_settings.debug:
            # 调试模式：在调用CUDA前复制参数到CPU，以便错误时保存快照
            cpu_args = cpu_deep_copy_tuple(args)  # 深拷贝到CPU
            try:
                # 调用C++/CUDA扩展的反向传播函数
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                # 如果发生错误，保存参数快照用于调试
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            # 正常模式：直接调用CUDA反向传播函数
            # 返回所有参数的梯度：
            # - grad_means2D: 2D投影点的梯度，形状: (N, 4)
            # - grad_colors_precomp: 预计算颜色的梯度，形状: (N, 3)
            # - grad_opacities: 不透明度的梯度，形状: (N, 1)
            # - grad_means3D: 3D位置的梯度，形状: (N, 3)
            # - grad_cov3Ds_precomp: 协方差矩阵的梯度，形状: (N, 6)
            # - grad_dc: 球谐DC分量的梯度，形状: (N, 1, 3)
            # - grad_sh: 球谐其他分量的梯度，形状: (N, K, 3)
            # - grad_scales: 缩放参数的梯度，形状: (N, 3)
            # - grad_rotations: 旋转参数的梯度，形状: (N, 4)
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # ============ 组织梯度元组 ============
        # 按照forward函数的输入参数顺序返回梯度
        # 注意：返回顺序必须与forward的输入参数完全对应
        grads = (
            grad_means3D,        # 对应means3D
            grad_means2D,        # 对应means2D
            grad_dc,             # 对应dc
            grad_sh,             # 对应sh
            grad_colors_precomp, # 对应colors_precomp
            grad_opacities,      # 对应opacities
            grad_scales,         # 对应scales
            grad_rotations,      # 对应rotations
            grad_cov3Ds_precomp, # 对应cov3Ds_precomp
            None,                # 对应raster_settings（配置对象不需要梯度）
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    mult : float
    prefiltered : bool
    debug : bool
    get_flag : bool
    metric_map : torch.Tensor

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, dc = None, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.Tensor([])
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings
        )

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)