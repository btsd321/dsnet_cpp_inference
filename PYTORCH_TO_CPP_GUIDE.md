# PyTorch模型到C++推理完整指南

本指南详细说明如何将训练好的PyTorch DSNet模型转换为可在C++中使用的推理库。

## 1. 环境准备

### 1.1 依赖库安装

```bash
# 安装LibTorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
export LibTorch_DIR=/path/to/libtorch

# 如果需要CUDA支持
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip

# 安装PCL和其他依赖
sudo apt-get install libpcl-dev libeigen3-dev
```

### 1.2 编译配置

```bash
cd dsnet_cpp_inference
mkdir build && cd build

# 基本编译（仅CPU）
cmake .. -DUSE_LIBTORCH=ON -DLibTorch_DIR=/path/to/libtorch

# 启用CUDA支持
cmake .. -DUSE_LIBTORCH=ON -DUSE_CUDA=ON -DLibTorch_DIR=/path/to/libtorch

# 启用OpenCV可视化
cmake .. -DUSE_LIBTORCH=ON -DUSE_OPENCV=ON -DLibTorch_DIR=/path/to/libtorch

make -j8
```

## 2. 模型转换流程

### 2.1 PyTorch模型转换为TorchScript

使用提供的转换脚本：

```python
python convert_model_to_torchscript.py \
    --model_path /path/to/your/pytorch_model.pth \
    --output_path ./Data/model/dsnet_model.pt \
    --model_type trace \
    --input_size 16384 \
    --verify
```

### 2.2 转换参数说明

- `model_path`: PyTorch模型文件路径（.pth或.pt）
- `output_path`: 输出的TorchScript模型路径
- `model_type`: 转换方式（trace或script）
- `input_size`: 输入点云大小（默认16384）
- `verify`: 验证转换后模型的正确性

### 2.3 模型输入输出格式

**输入格式：**
- Tensor形状: `[batch_size, num_points, 3]`
- 数据类型: `float32`
- 坐标范围: 建议归一化到[-1, 1]

**输出格式：**
- Tensor形状: `[batch_size, num_points, 4]`
- 数据类型: `float32`
- 输出通道:
  - `[0]`: seal_score (密封评分)
  - `[1]`: wrench_score (扭矩评分) 
  - `[2]`: feasibility_score (可行性评分)
  - `[3]`: object_size_score (物体尺寸评分)

## 3. C++推理库使用

### 3.1 基本使用流程

```cpp
#include "dsnet_inference/dsnet_inference.h"

// 1. 配置推理参数
dsnet::InferenceConfig config;
config.num_points = 16384;
config.use_gpu = true;

// 2. 创建推理器
dsnet::DSNetInference inference("./Data/model/dsnet_model.pt", config);

// 3. 初始化
if (!inference.initialize()) {
    std::cerr << "初始化失败!" << std::endl;
    return -1;
}

// 4. 加载点云
auto cloud = dsnet::loadPointCloudFromFile("test.pcd");

// 5. 执行推理
auto result = inference.predict(cloud);

// 6. 获取结果
auto best_points = inference.getBestSuctionPoints(
    result.getAllScores(), cloud, 10);
```

### 3.2 推理配置选项

```cpp
dsnet::InferenceConfig config;

// 基本参数
config.num_points = 16384;        // 点云采样数量
config.diffusion_steps = 50;      // 扩散步数
config.use_gpu = true;            // 是否使用GPU
config.device = "cuda";           // 设备类型

// 评分权重设置
config.score_weights.seal_weight = 0.3f;        // 密封评分权重
config.score_weights.wrench_weight = 0.3f;      // 扭矩评分权重
config.score_weights.visibility_weight = 0.2f;  // 可见性评分权重
config.score_weights.collision_weight = 0.2f;   // 碰撞评分权重
```

### 3.3 结果分析

```cpp
// 获取统计信息
auto stats = result.getStatistics();
std::cout << "平均综合评分: " << stats.mean_composite_score << std::endl;
std::cout << "最高评分: " << stats.max_composite_score << std::endl;
std::cout << "推理时间: " << result.getInferenceTime() << " ms" << std::endl;

// 获取最佳吸取点
auto best_points = inference.getBestSuctionPoints(scores, cloud, 10);
for (const auto& point : best_points) {
    std::cout << "位置: (" << point.position.x << ", " 
              << point.position.y << ", " << point.position.z << ")" << std::endl;
    std::cout << "综合评分: " << point.scores.composite_score << std::endl;
}
```

### 3.4 结果保存

```cpp
// 保存为不同格式
result.saveToTXT("result.txt");      // 文本格式
result.saveToJSON("result.json");    // JSON格式  
result.saveToCSV("result.csv");      // CSV格式

// 可视化（需要OpenCV）
#ifdef USE_OPENCV
dsnet::visualizeResult(result, "visualization.png");
#endif
```

## 4. 性能优化建议

### 4.1 GPU加速

```cpp
// 确保CUDA可用
if (torch::cuda::is_available()) {
    config.use_gpu = true;
    config.device = "cuda";
} else {
    config.use_gpu = false;
    config.device = "cpu";
}
```

### 4.2 批量推理

```cpp
// 批量处理多个点云
std::vector<PointCloud::Ptr> clouds = {cloud1, cloud2, cloud3};
auto results = inference.predictBatch(clouds);
```

### 4.3 内存优化

```cpp
// 预分配内存，避免频繁分配
config.num_points = 16384;  // 固定点云大小
```

## 5. 常见问题与解决方案

### 5.1 模型加载失败

**问题:** `LibTorch模型加载失败`

**解决方案:**
1. 检查TorchScript模型文件是否存在
2. 确认LibTorch版本兼容性
3. 验证模型转换是否成功

```bash
# 验证TorchScript文件
python -c "import torch; model = torch.jit.load('dsnet_model.pt'); print('模型加载成功')"
```

### 5.2 推理结果异常

**问题:** 推理结果全为0或异常值

**解决方案:**
1. 检查输入点云格式和范围
2. 确认模型输入输出维度匹配
3. 验证点云预处理是否正确

```cpp
// 检查点云数据
std::cout << "点云大小: " << cloud->size() << std::endl;
std::cout << "点云范围: " << getPointCloudBoundingBox(cloud) << std::endl;
```

### 5.3 性能问题

**问题:** 推理速度较慢

**解决方案:**
1. 启用GPU加速
2. 优化点云采样数量
3. 使用批量推理

```cpp
// 性能监控
auto start = std::chrono::high_resolution_clock::now();
auto result = inference.predict(cloud);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "推理时间: " << duration.count() << " ms" << std::endl;
```

## 6. 完整示例

参考 `inference_example.cpp` 获取完整的使用示例。

## 7. API参考

详细的API文档请参考头文件注释：
- `dsnet_inference/dsnet_inference.h` - 主推理类
- `dsnet_inference/dsnet_inference_result.h` - 结果处理类
- `dsnet_inference/dsnet_utils.h` - 工具函数

## 8. 技术支持

如遇到问题，请检查：
1. 编译配置是否正确
2. 依赖库版本是否兼容
3. 输入数据格式是否符合要求
4. 模型转换是否成功
