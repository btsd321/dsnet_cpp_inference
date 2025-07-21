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

namespace dsnet
{

/**
 * @brief 3D点结构
 */
struct Point3D
{
    float x, y, z;

    Point3D() : x(0), y(0), z(0) {}
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // 转换为Eigen向量
    Eigen::Vector3f toEigen() const
    {
        return Eigen::Vector3f(x, y, z);
    }

    // 从Eigen向量创建
    static Point3D fromEigen(const Eigen::Vector3f& vec)
    {
        return Point3D(vec.x(), vec.y(), vec.z());
    }
};

/**
 * @brief 吸取评分结构
 */
struct SuctionScores
{
    float seal_score;         // 密封评分
    float wrench_score;       // 扭矩评分
    float feasibility_score;  // 可行性评分
    float object_size_score;  // 物体尺寸评分
    float composite_score;    // 综合评分

    SuctionScores()
        : seal_score(0.0f),
          wrench_score(0.0f),
          feasibility_score(0.0f),
          object_size_score(0.0f),
          composite_score(0.0f)
    {
    }
};

/**
 * @brief 吸取点结构
 */
struct SuctionPoint
{
    Point3D position;      // 3D位置
    Point3D normal;        // 法向量
    SuctionScores scores;  // 评分
    int index;             // 在点云中的索引

    SuctionPoint() : index(-1) {}
};

/**
 * @brief 点云结构
 */
class PointCloud
{
   public:
    std::vector<Point3D> points;   // 点坐标
    std::vector<Point3D> normals;  // 法向量（可选）

    PointCloud() = default;
    PointCloud(const std::vector<Point3D>& pts) : points(pts) {}
    PointCloud(const std::vector<Point3D>& pts, const std::vector<Point3D>& norms)
        : points(pts), normals(norms)
    {
    }

    // 获取点数
    size_t size() const
    {
        return points.size();
    }

    // 检查是否有法向量
    bool hasNormals() const
    {
        return !normals.empty() && normals.size() == points.size();
    }

    // 添加点
    void addPoint(const Point3D& point, const Point3D& normal = Point3D())
    {
        points.push_back(point);
        if (hasNormals() || normal.x != 0 || normal.y != 0 || normal.z != 0)
        {
            normals.push_back(normal);
        }
    }

    // 清空
    void clear()
    {
        points.clear();
        normals.clear();
    }

    // 从文件加载
    bool loadFromFile(const std::string& filename);

    // 保存到文件
    bool saveToFile(const std::string& filename) const;

    // 归一化到单位球
    void normalize();

    // 计算边界框
    std::pair<Point3D, Point3D> getBoundingBox() const;
};

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
        float seal_weight = 0.3f;
        float wrench_weight = 0.3f;
        float feasibility_weight = 0.3f;
        float object_size_weight = 0.1f;
    } score_weights;
};

/**
 * @brief DSNet推理结果
 */
struct InferenceResult
{
    std::vector<SuctionScores> scores;      // 所有点的评分
    std::vector<SuctionPoint> best_points;  // 最佳吸取点
    PointCloud preprocessed_cloud;          // 预处理后的点云
    float inference_time_ms;                // 推理时间（毫秒）

    InferenceResult() : inference_time_ms(0.0f) {}
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
    InferenceResult predict(const PointCloud& input_cloud);

    /**
     * @brief 批量推理
     * @param input_clouds 输入点云列表
     * @return 推理结果列表
     */
    std::vector<InferenceResult> predictBatch(const std::vector<PointCloud>& input_clouds);

    /**
     * @brief 获取最佳吸取点
     * @param scores 评分数组
     * @param cloud 点云
     * @param top_k 返回前k个点
     * @return 最佳吸取点列表
     */
    std::vector<SuctionPoint> getBestSuctionPoints(const std::vector<SuctionScores>& scores,
                                                   const PointCloud& cloud, int top_k = 10) const;

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
    PointCloud preprocessPointCloud(const PointCloud& input_cloud);

    /**
     * @brief 最远点采样
     * @param cloud 输入点云
     * @param num_samples 采样数量
     * @return 采样后的点云
     */
    PointCloud fpsSubsample(const PointCloud& cloud, int num_samples);

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
    std::vector<std::array<float, 4>> forwardInference(const PointCloud& preprocessed_cloud);
};

/**
 * @brief 工具函数：从文件加载点云
 * @param filename 文件名
 * @return 点云对象
 */
PointCloud loadPointCloudFromFile(const std::string& filename);

/**
 * @brief 工具函数：保存点云到文件
 * @param cloud 点云对象
 * @param filename 文件名
 * @return 是否成功
 */
bool savePointCloudToFile(const PointCloud& cloud, const std::string& filename);

/**
 * @brief 工具函数：保存推理结果
 * @param result 推理结果
 * @param filename 文件名
 * @return 是否成功
 */
bool saveInferenceResult(const InferenceResult& result, const std::string& filename);

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