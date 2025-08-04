# 扩散吸取网络推理系统使用指南

## 概述

本推理系统专门设计用于从原始图像数据（RGB图像、深度图像、掩码图像和相机信息）进行端到端的吸取点预测推理。

## 功能特性

- 集成了完整的数据预处理流程
- 支持从原始图像数据到吸取点预测的端到端推理
- 自动处理点云采样、归一化和法向量计算
- 输出详细的吸取评分和最佳吸取点推荐

## 使用方法

### 基本用法

```bash
python inference.py \
    --checkpoint /path/to/model/checkpoint.tar \
    --rgb rgb_image.png \
    --depth depth_image.png \
    --mask mask_image.png \
    --camera_info camera_info.yaml \
    --params parameter.json \
    --output results.npz \
    --device cuda \
    --num_points 16384 \
    --diffusion_steps 50 \
    --top_k 10
```

### 参数说明

**必需参数：**
- `--checkpoint`: 训练好的模型检查点文件路径
- `--rgb`: RGB图像文件路径
- `--depth`: 深度图像文件路径
- `--mask`: 掩码图像文件路径
- `--camera_info`: 相机信息文件路径

**可选参数：**
- `--params`: 参数配置文件路径 (默认使用 input/parameter.json)
- `--output`: 输出结果文件路径 (默认: ./inference_results.npz)
- `--device`: 计算设备 (cuda/cpu，默认: cuda)
- `--num_points`: 点云采样点数 (默认: 16384)
- `--diffusion_steps`: 扩散推理步数 (默认: 50)
- `--top_k`: 返回前k个最佳吸取点 (默认: 10)

## 输入文件要求

### 必需的输入文件：

1. **RGB图像** (.png): 标准彩色图像
2. **深度图像** (.png): 16位深度图像，像素值表示深度信息
3. **掩码图像** (.png): 灰度图像，白色(255)表示目标物体，其他为背景
4. **相机信息文件** (.yaml): 包含相机内参和外参的YAML文件

### 可选的配置文件：

5. **参数配置文件** (.json): 包含深度范围等参数的JSON文件

## 输出结果

推理系统会生成以下输出文件：

1. **主结果文件** (.npz): 包含所有点的吸取评分
   - `point_cloud`: 预处理后的点云坐标
   - `suction_seal_scores`: 密封评分
   - `suction_wrench_scores`: 扭矩评分
   - `suction_feasibility_scores`: 可行性评分
   - `object_size_scores`: 物体尺寸评分

2. **最佳点信息文件** (_best_points.json): 前k个最佳吸取点的详细信息
   - 包含位置坐标、各项评分和综合排名

## 快速开始

1. 运行示例脚本：
```bash
python run_inference_example.py
```

2. 使用提供的示例数据进行测试：
```bash
python inference.py \
    --checkpoint input/model/checkpoint.tar \
    --rgb input/rgb_images/test_bgr_30-30.png \
    --depth input/depth_images/test_depth_30-30.png \
    --mask input/segment_single_images/test_segment_30-30-26.png \
    --camera_info input/camera_info.yaml \
    --params input/parameter.json \
    --output output/test_results.npz
```

## 依赖项

确保安装以下依赖：
- torch
- numpy
- opencv-python
- open3d
- PyYAML
- pointnet2_ops_lib

## 注意事项

1. 确保所有输入文件路径正确且文件存在
2. 模型检查点文件必须与当前模型架构兼容
3. GPU推理需要足够的显存支持
4. 深度图像的格式和深度范围需要与参数配置文件匹配

## 故障排除

### 常见问题：

1. **导入模块失败**：检查所有依赖库是否正确安装
2. **文件不存在**：确认输入文件路径正确
3. **显存不足**：尝试使用CPU模式或减少点云数量
4. **预处理失败**：检查图像格式和相机信息文件格式

### 调试建议：

- 使用 `--device cpu` 进行调试
- 检查输入图像是否正确加载
- 验证相机信息文件格式
- 确保掩码图像正确标识目标物体
