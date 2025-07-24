# DSNet C++ 手写推理实现指南

由于 PointNet++ 中包含自定义 CUDA 操作，无法直接将 checkpoint.tar 转换为 LibTorch 可用的模型。因此，我们提供了完整的手写 C++ 推理实现。

## 🚀 实现方案概述

我们的手写实现包含以下关键组件：

### 1. 权重加载系统
- **ModelLoader**: 从导出的权重文件加载模型参数
- **权重结构**: 支持卷积层、归一化层、MLP网络等各种权重类型

### 2. PointNet++ 骨干网络
- **FarthestPointSampling**: 最远点采样算法的 C++ 实现
- **BallQuery**: 球查询算法的 C++ 实现  
- **SetAbstractionLayer**: Set Abstraction 层的完整实现
- **FeaturePropagationLayer**: Feature Propagation 层的完整实现

### 3. 扩散模型
- **DDIMScheduler**: DDIM 调度器的 C++ 实现
- **ScheduledCNNRefine**: 扩散去噪网络的 C++ 实现
- **AttentionModule**: 注意力机制的 C++ 实现
- **DDIMPipeline**: 完整的 DDIM 采样管道

## 📋 使用步骤

### 步骤 1: 导出权重文件

首先，使用 Python 脚本从 checkpoint.tar 中导出权重：

```bash
cd pyinference/example
python export_weights.py --checkpoint input/model/checkpoint.tar --output exported_weights
```

这将创建以下目录结构：
```
exported_weights/
├── model_config.json          # 模型配置
├── export_info.json          # 导出信息
├── pointnet/                 # PointNet++权重
│   ├── weights_info.json
│   ├── *.bin                 # 二进制权重文件
│   └── *.txt                 # 文本权重文件（调试用）
└── diffusion/                # 扩散模型权重
    ├── weights_info.json
    ├── *.bin
    └── *.txt
```

### 步骤 2: 编译 C++ 库

```bash
mkdir build && cd build
cmake .. -DUSE_LIBTORCH=OFF  # 关闭LibTorch，使用手写实现
make -j$(nproc)
```

### 步骤 3: 使用手写推理

```cpp
#include "dsnet_inference.h"
using namespace dsnet;

int main() {
    // 配置推理参数
    InferenceConfig config;
    config.num_points = 16384;
    config.diffusion_steps = 50;
    config.use_handwritten_impl = true;  // 使用手写实现
    
    // 创建推理器，传入导出的权重目录路径
    DSNetInference inferencer("exported_weights", config);
    
    // 初始化
    if (!inferencer.initialize()) {
        std::cerr << "推理器初始化失败" << std::endl;
        return -1;
    }
    
    // 加载点云
    PointCloud::Ptr cloud = loadPointCloudFromFile("input.txt");
    
    // 执行推理
    auto result = inferencer.predict(cloud);
    
    // 获取最佳吸取点
    auto best_points = result.getBestPoints();
    std::cout << "找到 " << best_points.size() << " 个最佳吸取点" << std::endl;
    
    return 0;
}
```

## 🔧 实现细节

### PointNet++ 实现要点

1. **最远点采样 (FPS)**
   - 使用贪心算法逐步选择最远的点
   - 时间复杂度: O(n²m)，其中 n 是输入点数，m 是采样点数

2. **球查询 (Ball Query)**
   - 在指定半径内查找邻域点
   - 限制每个球内的最大点数以控制计算量

3. **Set Abstraction**
   - 多尺度特征提取
   - 支持不同半径和采样数的组合

### 扩散模型实现要点

1. **DDIM 调度器**
   - 实现确定性去噪过程
   - 支持可配置的推理步数

2. **注意力机制**
   - 通道注意力：全局平均池化 + MLP
   - 空间注意力：跨通道统计 + 卷积

3. **噪声预测网络**
   - 时间步嵌入 + 噪声嵌入 + 特征融合
   - 残差连接确保稳定训练

## ⚡ 性能优化

### 建议的优化策略

1. **内存优化**
   - 使用 Eigen 的块操作减少内存分配
   - 预分配缓冲区重用内存

2. **计算优化**
   - 使用 OpenMP 并行化点云处理
   - 向量化操作提升数值计算效率

3. **缓存优化**
   - 预计算常用的查找表
   - 局部性优化减少内存访问

### 示例并行化代码

```cpp
// 在球查询中使用 OpenMP
#pragma omp parallel for
for (int i = 0; i < query_points.rows(); ++i) {
    // 并行处理每个查询点
    std::vector<int> neighbors = findNeighbors(query_points.row(i));
    // ...
}
```

## 🧪 测试和验证

### 单元测试
```bash
./bin/dsnet_test  # 运行所有测试
```

### 精度验证
建议与 Python 版本的输出进行对比：

```bash
# Python 推理
python inference.py --checkpoint checkpoint.tar --rgb input.jpg --depth depth.png --mask mask.png --camera_info camera.yaml

# C++ 推理  
./bin/dsnet_inference --weights exported_weights --input pointcloud.txt

# 比较结果
python compare_results.py python_output.npz cpp_output.csv
```

## 🔍 故障排除

### 常见问题

1. **权重加载失败**
   - 检查权重文件路径是否正确
   - 确认权重文件完整性

2. **精度差异**
   - 检查数值类型匹配 (float32)
   - 验证算法实现的一致性

3. **性能问题**
   - 启用编译器优化 (-O3)
   - 检查内存访问模式

### 调试工具

```cpp
// 启用详细日志
config.verbose = true;

// 保存中间结果
result.saveDebugInfo("debug_output/");

// 检查数值范围
std::cout << "特征范围: [" << features.minCoeff() << ", " 
          << features.maxCoeff() << "]" << std::endl;
```

## 📈 路线图

### 待实现功能

1. **GPU 加速版本**
   - 使用 CUDA 或 OpenCL 实现关键算法
   - 内存管理优化

2. **模型量化**
   - INT8 量化支持
   - 模型压缩技术

3. **更多部署选项**
   - TensorRT 集成
   - 移动端优化

这个手写实现方案虽然工作量较大，但提供了完全的控制权和可定制性，能够确保在各种环境下的稳定运行。
