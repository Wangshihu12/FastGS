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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render_fastgs, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from fused_ssim import fused_ssim as fast_ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import random
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    """
    FastGS训练主函数，实现动态高斯喷绘的训练流程
    
    参数:
        dataset: 数据集配置参数，包含场景路径、背景颜色等信息
        opt: 优化参数，包含学习率、迭代次数、密集化参数等
        pipe: 渲染管线参数，控制渲染行为
        testing_iterations: 测试迭代次数列表，用于定期评估模型
        saving_iterations: 保存迭代次数列表，用于定期保存模型
    
    返回:
        无返回值，训练结果保存到磁盘
    """
    # 初始化TensorBoard记录器和输出目录
    tb_writer = prepare_output_and_logger(dataset)
    
    # 创建高斯模型，sh_degree控制球谐函数的阶数（用于表示颜色）
    gaussians = GaussianModel(dataset.sh_degree)
    
    # 创建变形网络模型，用于处理动态场景的时序变化
    # is_blender: 是否为Blender渲染数据，is_6dof: 是否使用6自由度变形
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    # 初始化场景，加载相机、图像等数据
    scene = Scene(dataset, gaussians)
    
    # 设置高斯模型的训练参数，初始化优化器
    gaussians.training_setup(opt, args)

    # 设置背景颜色：白色[1,1,1]或黑色[0,0,0]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # 将背景颜色转换为CUDA张量，形状: [3]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于记录迭代时间
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 创建CUDA事件用于记录优化器时间
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0  # 累计训练时间（秒）

    # 初始化视点栈和损失记录
    viewpoint_stack = None  # 存储训练相机视点的栈
    ema_loss_for_log = 0.0  # 指数移动平均损失，用于平滑显示
    # best_psnr = 0.0  # 记录最佳PSNR值（已注释）
    # best_iteration = 0  # 记录最佳迭代次数（已注释）
    
    # 创建进度条，显示训练进度
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    
    # 创建平滑噪声函数，用于时间扰动，随迭代次数衰减
    # lr_init=0.1: 初始噪声强度, lr_final=1e-15: 最终噪声强度
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    
    # ============ 主训练循环 ============
    for iteration in range(1, opt.iterations + 1):
        # 网络GUI交互部分：允许实时可视化训练过程
        if network_gui.conn == None:
            network_gui.try_connect()  # 尝试连接GUI客户端
        
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # 接收来自GUI的命令：自定义相机、训练控制等
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                
                if custom_cam != None:
                    # 渲染自定义视角的图像
                    net_image = render(custom_cam, gaussians, pipe, background, opt.mult, scaling_modifer)["render"]
                    # 将渲染结果转换为字节数组，形状: [H, W, 3]，范围: [0, 255]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                # 发送渲染图像到GUI
                network_gui.send(net_image_bytes, dataset.source_path)
                
                # 如果GUI要求继续训练，则跳出GUI循环
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                # 连接断开时重置
                network_gui.conn = None

        # 记录迭代开始时间
        iter_start.record()

        # 更新学习率（基于迭代次数的学习率调度）
        gaussians.update_learning_rate(iteration)
        deform.update_learning_rate(iteration)

        # 每1000次迭代增加球谐函数阶数，逐步提升颜色表达能力
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ============ 随机选择训练视点 ============
        # 如果视点栈为空，重新加载所有训练相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # 计算时间间隔，用于归一化帧ID
        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        # 随机弹出一个相机视点
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # 如果启用按需加载，将相机数据加载到GPU
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        
        # 获取当前帧的时间ID，形状: [1] 或标量
        fid = viewpoint_cam.fid

        # ============ 计算变形参数 ============
        if iteration < opt.warm_up:
            # 预热阶段：不应用变形，保持静态高斯
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            # 获取高斯点的数量
            N = gaussians.get_xyz.shape[0]  # N: 高斯点数量
            
            # 将帧ID扩展到所有高斯点，形状: [N, 1]
            time_input = fid.unsqueeze(0).expand(N, -1)

            # 添加时间扰动噪声（仅对非Blender数据）
            # ast_noise形状: [N, 1]，随机噪声用于增强时间连续性
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            
            # 通过变形网络计算高斯点的位移、旋转和缩放变化
            # d_xyz形状: [N, 3] - 位置偏移
            # d_rotation形状: [N, 4] - 旋转四元数偏移
            # d_scaling形状: [N, 3] - 缩放偏移
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # ============ 渲染图像 ============
        # 使用FastGS渲染器渲染当前视点
        render_pkg_re = render_fastgs(viewpoint_cam, gaussians, pipe, background, opt.mult, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        
        # 提取渲染结果
        # image形状: [3, H, W] - 渲染的RGB图像
        # viewspace_point_tensor形状: [N, 3] - 视图空间中的高斯点位置
        # visibility_filter形状: [N] - 布尔张量，标记可见的高斯点
        # radii形状: [N] - 每个高斯在屏幕空间的半径（像素）
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]  # 深度图（未使用）

        # ============ 计算损失 ============
        # 获取真实图像，形状: [3, H, W]
        gt_image = viewpoint_cam.original_image.cuda()
        
        # 计算L1损失
        Ll1 = l1_loss(image, gt_image)
        
        # 组合损失：L1损失 + SSIM损失（结构相似性）
        # lambda_dssim控制SSIM权重，unsqueeze(0)添加batch维度: [1, 3, H, W]
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
        
        # 反向传播计算梯度
        loss.backward()

        # 记录迭代结束时间
        iter_end.record()

        # 如果启用按需加载，释放相机数据到CPU
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        # ============ 训练统计和维护（不计算梯度） ============
        with torch.no_grad():
            # 更新指数移动平均损失，用于平滑显示
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            # 每10次迭代更新进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Gaussian Number": f"{gaussians._xyz.shape[0]:.{2}f}"})
                progress_bar.update(10)
            
            # 训练结束时关闭进度条
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录每个高斯点在屏幕空间的最大半径，用于后续剪枝
            # max_radii2D形状: [N]，记录历史最大半径
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # 计算迭代时间（毫秒）
            iter_time = iter_start.elapsed_time(iter_end)
            
            # 以下为测试和PSNR记录代码（已注释）
            # cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time,
            #                            testing_iterations, scene, render_fastgs, (pipe, background, opt.mult), deform,
            #                            dataset.load2gpu_on_the_fly, dataset.is_6dof)
            # if iteration in testing_iterations:
            #     if cur_psnr.item() > best_psnr:
            #         best_psnr = cur_psnr.item()
            #         best_iteration = iteration

            # ============ 模型保存 ============
            # 在指定迭代次数保存模型权重
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)  # 保存高斯模型
                deform.save_weights(args.model_path, iteration)  # 保存变形网络

            # ============ 高斯点密集化（Densification） ============
            optim_start.record()  # 记录优化开始时间
            
            # 在指定迭代范围内进行密集化
            if iteration < opt.densify_until_iter:
                # 累积密集化统计信息：位置梯度和可见性
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 定期执行密集化和剪枝
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 设置屏幕空间大小阈值
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # 获取训练相机列表并采样
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    # FastGS的多视图一致性密集化：计算每个高斯点的重要性和剪枝分数
                    # importance_score形状: [N] - 高斯点重要性分数
                    # pruning_score形状: [N] - 高斯点剪枝分数
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, background, opt, d_xyz, d_rotation, d_scaling, dataset.is_6dof, DENSIFY=True)
                    
                    # 执行密集化和剪枝操作
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold,  # 最大屏幕空间尺寸
                                                min_opacity = 0.005,  # 最小不透明度阈值
                                                extent = scene.cameras_extent,  # 场景范围
                                                radii=radii,  # 当前帧的半径
                                                args = opt,
                                                importance_score = importance_score,  # 重要性分数
                                                pruning_score = pruning_score)  # 剪枝分数

                # 定期重置不透明度，防止过度优化
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # ============ 最终剪枝阶段 ============
            # 在15000-30000迭代之间，每3000次迭代执行一次更激进的剪枝
            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                # 重新采样相机
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                # 计算剪枝分数（不需要重要性分数）
                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, background, opt, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
                
                # 执行最终剪枝，使用更高的不透明度阈值
                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
            
            # ============ 优化器更新 ============
            # 在最后一次迭代之前，执行优化器步骤
            if iteration < opt.iterations:
                # 更新变形网络参数
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                
                # 更新高斯模型参数（自适应密度控制）
                gaussians.optimizer_step(iteration)

            # 记录优化结束时间
            optim_end.record()
            torch.cuda.synchronize()  # 同步CUDA操作
            
            # 计算优化时间并累加到总时间（转换为秒）
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    # ============ 训练完成，输出统计信息 ============
    # print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))  # 已注释
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")  # 输出最终高斯点数量
    print(f"Dash time: {total_time}")  # 输出总训练时间（秒）


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30000,40000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000,40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
