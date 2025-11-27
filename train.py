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
import numpy as np
import os, random, time
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    """
    FastGS（Fast Gaussian Splatting）训练主函数
    
    功能描述：
        执行3D高斯溅射模型的训练过程，包括渲染、损失计算、密集化和剪枝等操作。
        该函数实现了FastGS的多视图一致性密集化和剪枝策略。
    
    参数说明：
        dataset: 数据集参数对象，包含场景数据、相机参数等配置
        opt: 优化参数对象，包含学习率、迭代次数、密集化参数等
        pipe: 渲染管线参数对象，包含渲染相关配置
        testing_iterations: 测试迭代列表，指定在哪些迭代进行测试评估
        saving_iterations: 保存迭代列表，指定在哪些迭代保存模型
        checkpoint_iterations: 检查点迭代列表，指定在哪些迭代保存检查点
        checkpoint: 检查点文件路径，用于恢复训练（可选）
        debug_from: 调试起始迭代，从该迭代开始启用调试模式
        websockets: 是否启用WebSocket进行实时可视化
    
    返回值：
        无返回值，训练完成后会打印高斯点数量和训练时间
    """
    # ============ 初始化阶段 ============
    first_iter = 0  # 起始迭代编号
    tb_writer = prepare_output_and_logger(dataset)  # 准备TensorBoard日志记录器
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)  # 创建高斯模型，sh_degree为球谐函数的阶数
    scene = Scene(dataset, gaussians)  # 创建场景对象，加载相机和图像数据
    gaussians.training_setup(opt)  # 设置优化器和学习率调度器
    
    # 如果提供了检查点，则恢复模型和迭代编号
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)  # 加载模型参数和迭代编号
        gaussians.restore(model_params, opt)  # 恢复高斯模型的参数

    # 设置背景颜色：白色背景为[1,1,1]，黑色背景为[0,0,0]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 背景颜色张量，形状: (3,)

    # 创建CUDA事件用于计时（用于测量单次迭代的渲染时间）
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 复制训练相机列表，用于随机采样视角
    viewpoint_stack = scene.getTrainCameras().copy()  # 相机列表
    viewpoint_indices = list(range(len(viewpoint_stack)))  # 相机索引列表

    # 创建CUDA事件用于计时优化步骤（包括密集化、剪枝、参数更新）
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0  # 累计训练时间（秒）

    ema_loss_for_log = 0.0  # 用于进度条显示的指数移动平均损失
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")  # 进度条
    first_iter += 1  # 从下一次迭代开始（避免重复）
    
    # 如果启用随机背景，则每次迭代使用随机颜色；否则使用固定背景
    bg = torch.rand((3), device="cuda") if opt.random_background else background  # 背景张量，形状: (3,)
    img_num = -1  # 记录训练图像数量

    # ============ 训练主循环 ============
    for iteration in range(first_iter, opt.iterations + 1):

        # -------- WebSocket实时可视化 --------
        if websockets:
            # 如果用户通过GUI选择了某个相机视角，则渲染该视角并发送到前端
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]  # 获取用户选择的相机
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]  # 渲染图像，形状: (3, H, W)
                network_gui_ws.latest_width = cam.image_width  # 更新图像宽度
                network_gui_ws.latest_height = cam.image_height  # 更新图像高度
                # 将渲染结果转换为字节流并发送到前端（将范围[0,1]映射到[0,255]，并转换为HWC格式）
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()  # 开始计时本次迭代的渲染时间
        
        # -------- 更新学习率 --------
        gaussians.update_learning_rate(iteration)  # 根据迭代次数动态调整学习率

        # -------- 每1000次迭代提升球谐函数阶数 --------
        # 球谐函数用于表示高斯的颜色，阶数越高表示能力越强
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()  # 增加球谐函数的阶数（最高到设定的最大阶数）

        # -------- 随机选择一个训练视角 --------
        # 如果相机栈为空，则重新填充（实现epoch的概念）
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()  # 重新复制所有训练相机
            viewpoint_indices = list(range(len(viewpoint_stack)))  # 重置索引
            if img_num == -1:
                img_num = len(viewpoint_stack)  # 记录训练图像总数
        rand_idx = randint(0, len(viewpoint_indices) - 1)  # 随机选择一个索引
        viewpoint_cam = viewpoint_stack.pop(rand_idx)  # 从栈中取出该相机（避免重复采样）
        _ = viewpoint_indices.pop(rand_idx)  # 同步删除索引

        # -------- 启用调试模式 --------
        if (iteration - 1) == debug_from:
            pipe.debug = True  # 从指定迭代开始启用调试模式

        # -------- 渲染当前视角 --------
        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)  # 使用FastGS渲染管线渲染图像
        # 解包渲染结果：
        # - image: 渲染的RGB图像，形状: (3, H, W)
        # - viewspace_point_tensor: 视图空间中的高斯点坐标（用于计算梯度），形状: (N, 3)
        # - visibility_filter: 可见性过滤器（标记哪些高斯点在该视角可见），形状: (N,)，布尔类型
        # - radii: 每个高斯点在图像空间的半径（像素单位），形状: (N,)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # -------- 计算损失并反向传播 --------
        gt_image = viewpoint_cam.original_image.cuda()  # 加载真实图像到GPU，形状: (3, H, W)
        Ll1 = l1_loss(image, gt_image)  # 计算L1损失（逐像素绝对值差）
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))  # 计算SSIM（结构相似性），需要增加batch维度，形状变为 (1, 3, H, W)
        # 组合损失：L1损失 + D-SSIM损失（1-SSIM）
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()  # 反向传播，计算梯度

        iter_end.record()  # 结束计时本次迭代的渲染时间

        # -------- 无梯度模式下的后处理操作 --------
        with torch.no_grad():
            # ---- 更新进度条 ----
            # 使用指数移动平均平滑损失曲线（0.4权重给当前损失，0.6权重给历史平均）
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})  # 显示平滑后的损失
                progress_bar.update(10)  # 每10次迭代更新一次进度条
            if iteration == opt.iterations:
                progress_bar.close()  # 训练结束时关闭进度条

            iter_time = iter_start.elapsed_time(iter_end)  # 计算本次迭代的渲染时间（毫秒）
            
            # ---- 保存模型 ----
            # 如果当前迭代在保存列表中，则保存高斯模型
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)  # 保存高斯点云和相机参数
            
            optim_start.record()  # 开始计时优化步骤
            
            # ============ 密集化阶段（Densification） ============
            # 仅在指定的迭代范围内进行密集化操作
            if iteration < opt.densify_until_iter:
                # ---- 更新最大半径统计 ----
                # 记录每个可见高斯点在图像空间的最大半径（用于后续剪枝）
                # gaussians.max_radii2D: 形状 (N,)，存储每个高斯点的历史最大半径
                # visibility_filter: 形状 (N,)，布尔张量，标记当前视角可见的高斯点
                # radii[visibility_filter]: 形状 (M,)，当前视角可见高斯点的半径
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                # ---- 累积密集化统计信息 ----
                # 累积视图空间梯度和可见性信息，用于判断哪些区域需要增加高斯点
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # ---- 执行密集化和剪枝操作 ----
                # 每隔一定迭代间隔，在满足条件时进行密集化
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 设置尺寸阈值：在不透明度重置后使用20像素，之前不限制
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()  # 获取所有训练相机
                    camlist = sampling_cameras(my_viewpoint_stack)  # 采样多个相机视角用于计算一致性分数

                    # FastGS的多视图一致性密集化：计算重要性分数和剪枝分数
                    # importance_score: 形状 (N,)，表示每个高斯点的重要性（需要保留或分裂的程度）
                    # pruning_score: 形状 (N,)，表示每个高斯点的剪枝分数（应该被剪枝的程度）
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    
                    # 执行FastGS的密集化和剪枝操作
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold,  # 最大屏幕空间尺寸阈值
                                                min_opacity = 0.005,  # 最小不透明度阈值（低于此值的高斯点会被剪枝）
                                                extent = scene.cameras_extent,  # 场景范围（用于判断高斯点是否太大）
                                                radii=radii,  # 当前迭代各高斯点的半径
                                                args = opt,  # 优化参数
                                                importance_score = importance_score,  # 重要性分数
                                                pruning_score = pruning_score)  # 剪枝分数

                # ---- 重置不透明度 ----
                # 定期重置高斯点的不透明度，防止所有点的不透明度都变得很高
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()  # 将不透明度重置为较低的值

            # ============ FastGS多视图一致性剪枝阶段 ============
            # 在15k到30k迭代之间，每3000次迭代执行一次更激进的剪枝
            # 此时模型基本收敛，可以更激进地剪枝而不影响渲染质量
            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                my_viewpoint_stack = scene.getTrainCameras().copy()  # 获取所有训练相机
                camlist = sampling_cameras(my_viewpoint_stack)  # 采样多个相机视角

                # 计算剪枝分数（不需要重要性分数，因为这里只做剪枝）
                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                
                # 执行最终剪枝操作（使用更高的不透明度阈值0.1）
                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
        
            # ============ 优化器步骤 ============
            # 更新高斯模型的参数（位置、旋转、缩放、不透明度、颜色等）
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    # 使用默认优化器（通常是Adam）
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    # 使用稀疏Adam优化器（仅更新可见的高斯点，提高效率）
                    visible = radii > 0  # 可见性标记：半径大于0表示在图像中可见，形状: (N,)
                    gaussians.optimizer.step(visible, radii.shape[0])  # 执行稀疏优化步骤
                    gaussians.optimizer.zero_grad(set_to_none = True)  # 清空梯度（set_to_none=True可以节省内存）

            # ---- 记录优化时间 ----
            optim_end.record()  # 结束计时优化步骤
            torch.cuda.synchronize()  # 同步CUDA，确保所有操作完成
            optim_time = optim_start.elapsed_time(optim_end)  # 计算优化时间（毫秒）
            total_time += (iter_time + optim_time) / 1e3  # 累积总时间（转换为秒）

    # ============ 训练结束 ============
    # 打印最终统计信息
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")  # 打印最终高斯点数量，_xyz形状: (N, 3)
    print(f"Training time: {total_time}")  # 打印总训练时间（秒）
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets
    )

    # All done
    print("\nTraining complete.")
