#pragma once

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#ifdef USE_LIBTORCH
#include <torch/torch.h>
#endif

#include <Eigen/Dense>

#include "dsnet_inference/dsnet_inference_result.h"
#include "dsnet_inference/dsnet_utils.h"

namespace dsnet
{

/**
 * @brief 推理配置参数
 */
struct InferenceConfig
{
    int num_points = 16384;       // 采样点数
    int diffusion_steps = 50;     // 扩散步数
    bool use_gpu = true;          // 是否使用GPU
    std::string device = "cuda";  // 设备类型

    // 评分权重
    struct ScoreWeights
    {
        // 密封评分权重
        float seal_weight = 0.25f;
        // 扭矩评分权重
        float wrench_weight = 0.25f;
        // 可见性评分权重
        float visibility_weight = 0.25f;
        // 碰撞评分权重
        float collision_weight = 0.25f;
    } score_weights;
};

/**
 * @brief DSNet C++推理器主类
 */
class DSNetInference
{
   public:
    /**
     * @brief 构造函数
     * @param model_path 模型文件路径
     * @param config 推理配置
     */
    DSNetInference(const std::string& model_path,
                   const InferenceConfig& config = InferenceConfig());

    /**
     * @brief 析构函数
     */
    ~DSNetInference();

    /**
     * @brief 初始化推理器
     * @return 是否成功
     */
    bool initialize();

    /**
     * @brief 执行推理
     * @param input_cloud 输入点云
     * @return 推理结果
     */
    InferenceResult predict(const PointCloud::Ptr& input_cloud);

    /**
     * @brief 批量推理
     * @param input_clouds 输入点云列表
     * @return 推理结果列表
     */
    std::vector<InferenceResult> predictBatch(const std::vector<PointCloud::Ptr>& input_clouds);

    /**
     * @brief 获取最佳吸取点
     * @param scores 评分数组
     * @param cloud 点云
     * @param top_k 返回前k个点
     * @return 最佳吸取点列表
     */
    std::vector<SuctionPoint> getBestSuctionPoints(const std::vector<SuctionScores>& scores,
                                                   const PointCloud::Ptr& cloud,
                                                   int top_k = 10) const;

    /**
     * @brief 设置配置
     * @param config 新配置
     */
    void setConfig(const InferenceConfig& config)
    {
        config_ = config;
    }

    /**
     * @brief 获取配置
     * @return 当前配置
     */
    const InferenceConfig& getConfig() const
    {
        return config_;
    }

    /**
     * @brief 检查是否已初始化
     * @return 是否已初始化
     */
    bool isInitialized() const
    {
        return initialized_;
    }

   private:
    std::string model_path_;  // 模型路径
    InferenceConfig config_;  // 配置参数
    bool initialized_;        // 是否已初始化

#ifdef USE_LIBTORCH
    torch::jit::script::Module model_;  // LibTorch模型
#endif

    /**
     * @brief 预处理点云
     * @param input_cloud 输入点云
     * @return 预处理后的点云
     */
    PointCloud::Ptr preprocessPointCloud(const PointCloud::Ptr& input_cloud);

    /**
     * @brief 最远点采样
     * @param cloud 输入点云
     * @param num_samples 采样数量
     * @return 采样后的点云
     */
    PointCloud::Ptr fpsSubsample(const PointCloud::Ptr& cloud, int num_samples);

    /**
     * @brief 计算综合评分
     * @param scores 各项评分
     * @return 综合评分
     */
    float computeCompositeScore(const SuctionScores& scores) const;

    /**
     * @brief 加载模型
     * @return 是否成功
     */
    bool loadModel();

    /**
     * @brief 执行前向推理
     * @param preprocessed_cloud 预处理后的点云
     * @return 原始推理结果
     */
    std::vector<std::array<float, 4>> forwardInference(const PointCloud::Ptr& preprocessed_cloud);
};

/**
 * @brief 工具函数：从文件加载点云
 * @param filename 文件名
 * @return 点云对象
 */
PointCloud::Ptr loadPointCloudFromFile(const std::string& filename);

/**
 * @brief 工具函数：保存点云到文件
 * @param cloud 点云对象
 * @param filename 文件名
 * @return 是否成功
 */
bool savePointCloudToFile(const PointCloud::Ptr& cloud, const std::string& filename);

/**
 * @brief 工具函数：保存推理结果
 * @param result 推理结果
 * @param filename 文件名
 * @return 是否成功
 */
bool saveInferenceResult(const InferenceResult& result, const std::string& filename);

/**
 * @brief 工具函数：归一化点云到单位球
 * @param cloud 输入点云
 */
void normalizePointCloud(PointCloud::Ptr& cloud);

/**
 * @brief 工具函数：计算点云边界框
 * @param cloud 输入点云
 * @return 最小点和最大点
 */
std::pair<Point3D, Point3D> getPointCloudBoundingBox(const PointCloud::Ptr& cloud);

/**
 * @brief 工具函数：可视化结果（需要OpenCV）
 * @param result 推理结果
 * @param output_path 输出图片路径
 * @return 是否成功
 */
#ifdef USE_OPENCV
bool visualizeResult(const InferenceResult& result, const std::string& output_path);
#endif

}  // namespace dsnet