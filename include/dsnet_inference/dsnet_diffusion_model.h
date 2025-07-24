#pragma once

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <vector>

#include "dsnet_model_loader.h"

namespace dsnet
{

/**
 * @brief 简单的MLP层实现
 */
class MLPLayer
{
   public:
    ConvWeights conv_;
    NormWeights norm_;

    MLPLayer() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input)
    {
        // TODO: Implement MLP forward pass
        return input;
    }
};

/**
 * @brief 注意力机制实现
 */
class AttentionModule
{
   private:
    ConvWeights channel_fc1_;
    ConvWeights channel_fc2_;
    ConvWeights spatial_conv_;

   public:
    AttentionModule(const ConvWeights& channel_fc1, const ConvWeights& channel_fc2,
                    const ConvWeights& spatial_conv);

    /**
     * @brief 通道注意力
     * @param input 输入特征 [C, N]
     * @return 注意力加权后的特征 [C, N]
     */
    Eigen::MatrixXf channelAttention(const Eigen::MatrixXf& input) const;

    /**
     * @brief 空间注意力
     * @param input 输入特征 [C, N]
     * @return 注意力加权后的特征 [C, N]
     */
    Eigen::MatrixXf spatialAttention(const Eigen::MatrixXf& input) const;

   private:
    Eigen::MatrixXf adaptiveAvgPool1d(const Eigen::MatrixXf& input) const;
    Eigen::MatrixXf adaptiveMaxPool1d(const Eigen::MatrixXf& input) const;
    Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& input) const;
};

/**
 * @brief DDIM调度器实现
 */
class DDIMScheduler
{
   private:
    int num_train_timesteps_;
    std::vector<float> betas_;
    std::vector<float> alphas_;
    std::vector<float> alphas_cumprod_;
    std::vector<int> timesteps_;
    bool clip_sample_;

   public:
    DDIMScheduler(int num_train_timesteps = 1000, bool clip_sample = false);

    /**
     * @brief 设置推理时间步
     * @param num_inference_steps 推理步数
     */
    void setTimesteps(int num_inference_steps);

    /**
     * @brief 添加噪声到干净图像
     * @param original_samples 原始样本 [B, N, C]
     * @param noise 噪声 [B, N, C]
     * @param timesteps 时间步 [B]
     * @return 加噪后的样本 [B, N, C]
     */
    Eigen::MatrixXf addNoise(const Eigen::MatrixXf& original_samples, const Eigen::MatrixXf& noise,
                             const std::vector<int>& timesteps) const;

    /**
     * @brief DDIM采样步骤
     * @param model_output 模型预测的噪声 [B, N, C]
     * @param timestep 当前时间步
     * @param sample 当前样本 [B, N, C]
     * @param eta 噪声系数
     * @param use_clipped_model_output 是否裁剪模型输出
     * @return 下一步的样本 [B, N, C]
     */
    Eigen::MatrixXf step(const Eigen::MatrixXf& model_output, int timestep,
                         const Eigen::MatrixXf& sample, float eta = 0.0f,
                         bool use_clipped_model_output = true) const;

    const std::vector<int>& getTimesteps() const
    {
        return timesteps_;
    }

   private:
    std::vector<float> linspace(float start, float end, int num) const;
    Eigen::MatrixXf clip(const Eigen::MatrixXf& input, float min_val, float max_val) const;
};

/**
 * @brief 扩散模型细化网络
 */
class ScheduledCNNRefine
{
   private:
    MLPLayer noise_embedding_;
    Eigen::MatrixXf time_embedding_;  // [1280, 128]
    MLPLayer prediction_network_;
    AttentionModule attention_;
    int channels_in_;
    int channels_noise_;

   public:
    ScheduledCNNRefine(const DiffusionWeights& weights, int channels_in = 128,
                       int channels_noise = 4);

    /**
     * @brief 前向传播
     * @param noisy_image 噪声图像 [B, N, C_noise]
     * @param timestep 时间步 [B] 或标量
     * @param features 条件特征 [B, N, C_in]
     * @return 预测噪声 [B, C_noise, N]
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& noisy_image, const std::vector<int>& timestep,
                            const Eigen::MatrixXf& features);

   private:
    Eigen::MatrixXf getTimeEmbedding(const std::vector<int>& timestep) const;
};

/**
 * @brief DDIM推理管道
 */
class DDIMPipeline
{
   private:
    std::unique_ptr<ScheduledCNNRefine> model_;
    std::unique_ptr<DDIMScheduler> scheduler_;
    std::mt19937 generator_;

   public:
    DDIMPipeline(std::unique_ptr<ScheduledCNNRefine> model,
                 std::unique_ptr<DDIMScheduler> scheduler);

    /**
     * @brief 执行DDIM采样
     * @param batch_size 批量大小
     * @param shape 输出形状 (不含batch维) [N, C]
     * @param features 条件特征 [B, N, C_feat]
     * @param num_inference_steps 推理步数
     * @param eta 噪声系数
     * @return 生成的样本 [B, N, C]
     */
    Eigen::MatrixXf generate(int batch_size, const std::pair<int, int>& shape,
                             const Eigen::MatrixXf& features, int num_inference_steps = 50,
                             float eta = 0.0f);

    void setSeed(unsigned int seed)
    {
        generator_.seed(seed);
    }

   private:
    Eigen::MatrixXf generateRandomNoise(int batch_size, const std::pair<int, int>& shape);
};

/**
 * @brief 完整的DSNet推理模型
 */
class DSNetModel
{
   private:
    std::unique_ptr<class PointNetBackbone> backbone_;
    std::unique_ptr<DDIMPipeline> pipeline_;
    ModelWeights::Config config_;
    float bit_scale_;

   public:
    DSNetModel(std::unique_ptr<ModelWeights> weights);

    /**
     * @brief 推理
     * @param point_clouds 输入点云 [B, N, 3]
     * @param normals 法向量 [B, N, 3]
     * @param num_inference_steps 扩散推理步数
     * @return 预测结果 [B, N, 4] (seal, wrench, feasibility, object_size)
     */
    Eigen::MatrixXf predict(const Eigen::MatrixXf& point_clouds, const Eigen::MatrixXf& normals,
                            int num_inference_steps = 20);

    const ModelWeights::Config& getConfig() const
    {
        return config_;
    }

   private:
    void buildModel(std::unique_ptr<ModelWeights> weights);
};

}  // namespace dsnet
