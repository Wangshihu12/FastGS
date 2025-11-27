/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/adam.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::function<int*(size_t N)> resizeIntFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return t.contiguous().data_ptr<int>();
    };
    return lambda;
}

std::function<float*(size_t N)> resizeFloatFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return t.contiguous().data_ptr<float>();
    };
    return lambda;
}

/**
 * [功能描述]：高斯溅射CUDA前向传播函数（PyTorch接口层）
 * 
 * 该函数是Python和C++/CUDA之间的桥梁，负责：
 * 1. 验证输入张量的合法性
 * 2. 准备输出缓冲区和中间缓冲区
 * 3. 将PyTorch张量的数据指针传递给底层CUDA光栅化器
 * 4. 调用核心CUDA光栅化实现
 * 5. 返回渲染结果和中间状态（用于反向传播）
 * 
 * @param background: 背景颜色张量，形状: (3,) - RGB三通道
 * @param means3D: 3D高斯中心位置，形状: (N, 3) - 每个高斯的xyz坐标
 * @param colors: 预计算的颜色（如果不使用球谐函数），形状: (N, 3)
 * @param opacity: 不透明度，形状: (N, 1) - 每个高斯的透明度
 * @param scales: 缩放参数，形状: (N, 3) - 每个高斯在xyz轴上的缩放
 * @param rotations: 旋转参数（四元数），形状: (N, 4) - 每个高斯的旋转
 * @param scale_modifier: 全局缩放修正因子 - 临时调整所有高斯的大小
 * @param cov3D_precomp: 预计算的3D协方差矩阵，形状: (N, 6) - 上三角矩阵展开
 * @param metric_map: 度量图，形状: (H*W,) - 用于记录每个像素的统计信息
 * @param viewmatrix: 视图矩阵，形状: (4, 4) - 世界坐标到相机坐标的变换
 * @param projmatrix: 投影矩阵，形状: (4, 4) - 相机坐标到屏幕坐标的变换
 * @param tan_fovx: 水平视场角的正切值（半角）
 * @param tan_fovy: 垂直视场角的正切值（半角）
 * @param image_height: 输出图像高度（像素）
 * @param image_width: 输出图像宽度（像素）
 * @param dc: 球谐函数的DC分量（第0阶），形状: (N, 1, 3)
 * @param sh: 球谐函数的其他分量（第1阶及以上），形状: (N, M, 3)
 * @param degree: 球谐函数的最大阶数
 * @param campos: 相机在世界坐标系中的位置，形状: (3,)
 * @param mult: FastGS的缩放乘数参数
 * @param prefiltered: 是否已预过滤（通常为false）
 * @param debug: 是否启用调试模式
 * @param get_flag: 是否获取度量统计信息
 * 
 * @return 返回包含9个元素的元组：
 *   - rendered (int): 实际渲染的高斯数量（可见的高斯）
 *   - num_buckets (int): 用于并行渲染的分桶数量
 *   - out_color (Tensor): 渲染的RGB图像，形状: (3, H, W)
 *   - radii (Tensor): 每个高斯的屏幕空间半径，形状: (N,)，单位：像素
 *   - geomBuffer (Tensor): 几何缓冲区（包含投影后的几何信息，用于反向传播）
 *   - binningBuffer (Tensor): 分桶缓冲区（包含排序信息，用于反向传播）
 *   - imgBuffer (Tensor): 图像缓冲区（包含渲染中间结果，用于反向传播）
 *   - sampleBuffer (Tensor): 采样缓冲区（FastGS特有，用于记录采样信息）
 *   - metricCount (Tensor): 度量计数，形状: (N,)
 */
std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& metric_map,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
    const float mult,
	const bool prefiltered,
	const bool debug,
	const bool get_flag)
{
  // ============ 输入验证 ============
  // 验证means3D张量的维度和形状是否正确
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  // ============ 提取基本参数 ============
  const int P = means3D.size(0);  // P: 高斯点的总数量
  const int H = image_height;     // H: 输出图像的高度（像素）
  const int W = image_width;      // W: 输出图像的宽度（像素）

  // 创建张量选项（用于后续创建新张量时指定数据类型和设备）
  auto int_opts = means3D.options().dtype(torch::kInt32);      // 整型张量选项
  auto float_opts = means3D.options().dtype(torch::kFloat32);  // 浮点型张量选项

  // ============ 准备输出张量 ============
  // 创建输出颜色图像张量，初始化为0
  // NUM_CHAFFELS通常为3（RGB），形状: (3, H, W)
  torch::Tensor out_color = torch::full({NUM_CHAFFELS, H, W}, 0.0, float_opts);
  
  // 创建半径张量，用于记录每个高斯在屏幕空间的半径（像素单位）
  // 初始化为0，形状: (P,)
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  // ============ 准备中间缓冲区 ============
  // 这些缓冲区用于在CUDA光栅化过程中存储中间结果，并在反向传播中使用
  torch::Device device(torch::kCUDA);         // 指定设备为CUDA
  torch::TensorOptions options(torch::kByte); // 字节类型选项（原始内存缓冲区）
  
  // 创建空的缓冲区张量（初始大小为0，将在CUDA内核中动态调整大小）
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));     // 几何缓冲区
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));  // 分桶缓冲区（用于tile排序）
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));      // 图像缓冲区
  torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));   // 采样缓冲区（FastGS特有）
  
  // 创建缓冲区调整大小的函数对象
  // 这些函数对象允许CUDA代码根据需要动态调整缓冲区大小
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);

  // ============ 准备度量统计信息 ============
  int* accum_metric_counts_ptr = nullptr;  // 累积度量计数指针（初始为空）
  torch::Tensor metricCount = torch::empty({0}, int_opts);  // 度量计数张量（初始为空）

  // 如果需要获取度量统计信息
  if(get_flag)
  {
	// 创建度量计数张量，初始化为0，形状: (P,)
	// 用于统计每个高斯对渲染的贡献次数
	metricCount = torch::full({P}, 0, int_opts);
	accum_metric_counts_ptr = metricCount.contiguous().data<int>();  // 获取数据指针
  }
  
  // ============ 执行CUDA光栅化 ============
  int rendered = 0;      // 实际渲染的高斯数量（初始化为0）
  int num_buckets = 0;   // 分桶数量（用于并行渲染，初始化为0）
  
  // 只有当存在高斯点时才执行光栅化
  if(P != 0)
  {
	  // 获取球谐函数的维度
	  int M = 0;  // M: 球谐系数的数量（不包括DC分量）
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);  // 获取球谐系数的第二维大小
      }

	  // 调用核心CUDA光栅化函数
	  // 该函数在cuda_rasterizer/rasterizer_impl.cu中实现
	  auto tup = CudaRasterizer::Rasterizer::forward(
	    // 缓冲区调整函数
	    geomFunc,      // 几何缓冲区调整函数
		binningFunc,   // 分桶缓冲区调整函数
		imgFunc,       // 图像缓冲区调整函数
		sampleFunc,    // 采样缓冲区调整函数
		
		// 基本参数
	    P,             // 高斯点数量
	    degree,        // 球谐函数阶数
	    M,             // 球谐系数数量
		
		// PyTorch张量数据指针（传递给CUDA内核）
		background.contiguous().data<float>(),          // 背景颜色指针，数据: float[3]
		W, H,                                           // 图像宽度和高度
		means3D.contiguous().data<float>(),             // 3D位置指针，数据: float[P][3]
		dc.contiguous().data_ptr<float>(),              // DC分量指针，数据: float[P][1][3]
		sh.contiguous().data_ptr<float>(),              // SH系数指针，数据: float[P][M][3]
		colors.contiguous().data<float>(),              // 预计算颜色指针，数据: float[P][3]
		opacity.contiguous().data<float>(),             // 不透明度指针，数据: float[P][1]
		scales.contiguous().data_ptr<float>(),          // 缩放指针，数据: float[P][3]
		scale_modifier,                                 // 全局缩放修正因子
		rotations.contiguous().data_ptr<float>(),       // 旋转指针（四元数），数据: float[P][4]
		cov3D_precomp.contiguous().data<float>(),       // 预计算协方差指针，数据: float[P][6]
		metric_map.contiguous().data<int>(),            // 度量图指针，数据: int[H*W]
		viewmatrix.contiguous().data<float>(),          // 视图矩阵指针，数据: float[4][4]
		projmatrix.contiguous().data<float>(),          // 投影矩阵指针，数据: float[4][4]
		campos.contiguous().data<float>(),              // 相机位置指针，数据: float[3]
        mult,                                           // FastGS缩放乘数
		tan_fovx,                                       // 水平视场角正切值
		tan_fovy,                                       // 垂直视场角正切值
		prefiltered,                                    // 是否预过滤
		out_color.contiguous().data<float>(),           // 输出颜色图像指针（原地修改），数据: float[3][H][W]
		radii.contiguous().data<int>(),                 // 输出半径指针（原地修改），数据: int[P]
		debug,                                          // 是否调试模式
		get_flag,                                       // 是否获取度量统计
		accum_metric_counts_ptr);                       // 度量计数指针（可选），数据: int[P]

	  // 从返回的元组中提取结果
	  rendered = std::get<0>(tup);      // 实际渲染的高斯数量
	  num_buckets = std::get<1>(tup);   // 分桶数量
  }
  
  // ============ 返回结果 ============
  // 返回包含渲染结果和中间状态的元组
  // 中间缓冲区（geomBuffer, binningBuffer, imgBuffer, sampleBuffer）将在反向传播中使用
  return std::make_tuple(rendered, num_buckets, out_color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, metricCount);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int B,
	const torch::Tensor& sampleBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());  // abs
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHAFFELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  dc.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_ddc.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

void adamUpdate(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
){
	ADAM::adamUpdate(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}