// #pragma once

// #include <Eigen/Dense>
// #include <map>
// #include <memory>
// #include <string>
// #include <vector>

// namespace dsnet
// {

// /**
//  * @brief 权重文件元信息
//  */
// struct WeightInfo
// {
//     std::vector<int> shape;
//     std::string filename;

//     WeightInfo() = default;
//     WeightInfo(const std::vector<int>& s, const std::string& f) : shape(s), filename(f) {}
// };

// /**
//  * @brief 卷积层权重
//  */
// struct ConvWeights
// {
//     Eigen::MatrixXf weight;  // 权重矩阵
//     Eigen::VectorXf bias;    // 偏置向量
//     bool has_bias = false;

//     ConvWeights() = default;
//     ConvWeights(int out_channels, int in_channels, bool bias = false)
//         : weight(out_channels, in_channels), has_bias(bias)
//     {
//         if (bias)
//         {
//             this->bias.resize(out_channels);
//         }
//     }
// };

// /**
//  * @brief BatchNorm/GroupNorm权重
//  */
// struct NormWeights
// {
//     Eigen::VectorXf weight;        // 缩放因子
//     Eigen::VectorXf bias;          // 偏移量
//     Eigen::VectorXf running_mean;  // 运行均值
//     Eigen::VectorXf running_var;   // 运行方差
//     float eps = 1e-5f;

//     NormWeights() = default;
//     NormWeights(int num_features)
//         : weight(num_features),
//           bias(num_features),
//           running_mean(num_features),
//           running_var(num_features)
//     {
//     }
// };

// /**
//  * @brief MLP网络权重
//  */
// struct MLPWeights
// {
//     std::vector<ConvWeights> conv_layers;
//     std::vector<NormWeights> norm_layers;
//     std::vector<bool> has_norm;  // 每层是否有归一化
// };

// /**
//  * @brief PointNet++骨干网络权重
//  */
// struct PointNetWeights
// {
//     // 每层的MLP权重
//     std::vector<MLPWeights> sa_mlps;  // Set Abstraction层
//     // 特征传播层权重
//     std::vector<MLPWeights> fp_mlps;  // Feature Propagation层
// };

// /**
//  * @brief 扩散模型权重
//  */
// struct DiffusionWeights
// {
//     // 噪声嵌入网络
//     MLPWeights noise_embedding;

//     // 时间步嵌入
//     Eigen::MatrixXf time_embedding;  // [1280, 128]

//     // 主预测网络
//     MLPWeights prediction_network;

//     // 注意力机制权重
//     ConvWeights channel_attention_fc1;
//     ConvWeights channel_attention_fc2;
//     ConvWeights spatial_attention_conv;
// };

// /**
//  * @brief 完整模型权重
//  */
// struct ModelWeights
// {
//     PointNetWeights backbone;
//     DiffusionWeights diffusion;

//     // 模型配置参数
//     struct Config
//     {
//         std::vector<int> npoint_per_layer = {4096, 1024, 256, 64};
//         std::vector<std::vector<float>> radius_per_layer = {
//             {10, 20, 30}, {30, 45, 60}, {60, 80, 120}, {120, 160, 240}};
//         int input_feature_dims = 3;
//         int backbone_feature_dim = 128;
//         int diffusion_steps = 20;
//         float bit_scale = 0.5f;
//     } config;
// };

// /**
//  * @brief 模型权重加载器
//  */
// class ModelLoader
// {
//    public:
//     /**
//      * @brief 从checkpoint.tar文件加载模型权重
//      * @param checkpoint_path checkpoint.tar文件路径
//      * @return 加载的模型权重
//      */
//     static std::unique_ptr<ModelWeights> loadFromCheckpoint(const std::string& checkpoint_path);

//     /**
//      * @brief 从Python脚本导出的权重文件加载
//      * @param weights_dir 权重文件目录
//      * @return 加载的模型权重
//      */
//     static std::unique_ptr<ModelWeights> loadFromExportedWeights(const std::string& weights_dir);

//     // Public helper functions for testing and external use
//     static bool loadBinaryTensor(const std::string& filePath, const std::vector<int>& shape,
//                                  Eigen::MatrixXf& tensor);
//     static bool loadTextTensor(const std::string& filePath, const std::vector<int>& shape,
//                                Eigen::MatrixXf& tensor);
//     static bool parseWeightMetadata(const std::string& weightsInfoPath,
//                                     std::map<std::string, WeightInfo>& weightMap);

//    private:
//     static bool loadModelConfig(const std::string& weights_dir, ModelWeights& weights);
//     static bool loadPointNetWeights(const std::string& weights_dir, PointNetWeights& weights);
//     static bool loadDiffusionWeights(const std::string& weights_dir, DiffusionWeights& weights);
//     static ConvWeights loadConvWeights(const std::string& prefix,
//                                        const std::map<std::string, std::vector<float>>&
//                                        state_dict);
//     static NormWeights loadNormWeights(const std::string& prefix,
//                                        const std::map<std::string, std::vector<float>>&
//                                        state_dict);
//     static MLPWeights loadMLPWeights(const std::string& prefix,
//                                      const std::map<std::string, std::vector<float>>& state_dict,
//                                      int num_layers);
// };

// }  // namespace dsnet
