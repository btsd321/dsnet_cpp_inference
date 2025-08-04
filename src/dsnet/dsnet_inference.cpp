// #include "dsnet_inference/dsnet_inference.h"

// #include <algorithm>
// #include <argparse/argparse.hpp>
// #include <chrono>
// #include <cmath>
// #include <fstream>
// #include <iostream>
// #include <random>
// #include <sstream>

// namespace dsnet
// {

// // ==================== DSNetInference Implementation ====================

// DSNetInference::DSNetInference(const std::string& model_path, const InferenceConfig& config)
//     : model_path_(model_path), config_(config), initialized_(false)
// {
// }

// DSNetInference::~DSNetInference()
// {
//     // 清理资源
// }

// bool DSNetInference::initialize()
// {
//     std::cout << "初始化 DSNet 推理器..." << std::endl;
//     std::cout << "模型路径: " << model_path_ << std::endl;
//     std::cout << "配置参数:" << std::endl;
//     std::cout << "  - 采样点数: " << config_.num_points << std::endl;
//     std::cout << "  - 扩散步数: " << config_.diffusion_steps << std::endl;
//     std::cout << "  - 设备: " << config_.device << std::endl;

//     // 检查模型文件是否存在
//     std::ifstream model_file(model_path_);
//     if (!model_file.good())
//     {
//         std::cerr << "错误: 模型文件不存在: " << model_path_ << std::endl;
//         return false;
//     }

//     // 加载模型
//     if (!loadModel())
//     {
//         std::cerr << "错误: 模型加载失败" << std::endl;
//         return false;
//     }

//     initialized_ = true;
//     std::cout << "DSNet 推理器初始化成功!" << std::endl;
//     return true;
// }

// bool DSNetInference::loadModel()
// {
// #ifdef USE_LIBTORCH
//     try
//     {
//         // 使用LibTorch加载模型
//         model_ = torch::jit::load(model_path_);
//         model_.eval();

//         if (config_.use_gpu && torch::cuda::is_available())
//         {
//             model_.to(torch::kCUDA);
//             std::cout << "模型已加载到GPU" << std::endl;
//         }
//         else
//         {
//             model_.to(torch::kCPU);
//             std::cout << "模型已加载到CPU" << std::endl;
//         }

//         return true;
//     }
//     catch (const std::exception& e)
//     {
//         std::cerr << "LibTorch模型加载失败: " << e.what() << std::endl;
//         return false;
//     }
// #else
//     // 没有LibTorch，使用占位符实现
//     std::cout << "警告: 未启用LibTorch，使用模拟推理" << std::endl;
//     return true;
// #endif
// }

// InferenceResult DSNetInference::predict(const PointCloud::Ptr& input_cloud)
// {
//     if (!initialized_)
//     {
//         throw std::runtime_error("推理器未初始化");
//     }

//     auto start_time = std::chrono::high_resolution_clock::now();

//     InferenceResult result;

//     // 预处理点云
//     auto preprocessed_cloud = preprocessPointCloud(input_cloud);
//     result.setPreprocessedCloud(preprocessed_cloud);

//     // 执行推理
//     auto raw_scores = forwardInference(preprocessed_cloud);

//     // 转换为SuctionScores格式
//     std::vector<SuctionScores> scores;
//     scores.resize(raw_scores.size());
//     for (size_t i = 0; i < raw_scores.size(); ++i)
//     {
//         scores[i].seal_score = raw_scores[i][0];
//         scores[i].wrench_score = raw_scores[i][1];
//         scores[i].visibility_score = raw_scores[i][2];  // 修正字段名
//         scores[i].collision_score = raw_scores[i][3];   // 修正字段名
//         scores[i].composite_score = computeCompositeScore(scores[i]);
//     }
//     result.setScores(scores);

//     // 获取最佳吸取点
//     auto best_points = getBestSuctionPoints(scores, preprocessed_cloud, 10);
//     result.setBestPoints(best_points);

//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
//     result.setInferenceTime(static_cast<float>(duration.count()));

//     std::cout << "推理完成，用时: " << result.getInferenceTime() << " ms" << std::endl;

//     return result;
// }

// std::vector<InferenceResult> DSNetInference::predictBatch(
//     const std::vector<PointCloud::Ptr>& input_clouds)
// {
//     std::vector<InferenceResult> results;
//     results.reserve(input_clouds.size());

//     for (const auto& cloud : input_clouds)
//     {
//         results.push_back(predict(cloud));
//     }

//     return results;
// }

// PointCloud::Ptr DSNetInference::preprocessPointCloud(const PointCloud::Ptr& input_cloud)
// {
//     std::cout << "预处理点云: " << input_cloud->points.size() << " 个点" << std::endl;

//     PointCloud::Ptr processed_cloud(new PointCloud(*input_cloud));

//     // 归一化
//     normalizePointCloud(processed_cloud);

//     // 采样到目标点数
//     if (static_cast<int>(processed_cloud->points.size()) != config_.num_points)
//     {
//         if (static_cast<int>(processed_cloud->points.size()) > config_.num_points)
//         {
//             // 下采样
//             processed_cloud = fpsSubsample(processed_cloud, config_.num_points);
//         }
//         else
//         {
//             // 上采样（重复采样）
//             PointCloud::Ptr upsampled_cloud(new PointCloud);
//             int repeat_times =
//                 (config_.num_points + static_cast<int>(processed_cloud->points.size()) - 1) /
//                 static_cast<int>(processed_cloud->points.size());

//             for (int r = 0; r < repeat_times &&
//                             static_cast<int>(upsampled_cloud->points.size()) <
//                             config_.num_points;
//                  ++r)
//             {
//                 for (size_t i = 0;
//                      i < processed_cloud->points.size() &&
//                      static_cast<int>(upsampled_cloud->points.size()) < config_.num_points;
//                      ++i)
//                 {
//                     upsampled_cloud->points.push_back(processed_cloud->points[i]);
//                 }
//             }

//             upsampled_cloud->width = upsampled_cloud->points.size();
//             upsampled_cloud->height = 1;
//             upsampled_cloud->is_dense = true;

//             processed_cloud = upsampled_cloud;
//         }
//     }

//     std::cout << "预处理后点云: " << processed_cloud->points.size() << " 个点" << std::endl;

//     return processed_cloud;
// }

// PointCloud::Ptr DSNetInference::fpsSubsample(const PointCloud::Ptr& cloud, int num_samples)
// {
//     if (static_cast<int>(cloud->points.size()) <= num_samples)
//     {
//         return PointCloud::Ptr(new PointCloud(*cloud));
//     }

//     std::vector<int> sampled_indices;
//     std::vector<float> distances(cloud->points.size(), std::numeric_limits<float>::max());

//     // 随机选择第一个点
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, static_cast<int>(cloud->points.size()) - 1);

//     int farthest = dis(gen);
//     sampled_indices.push_back(farthest);

//     for (int i = 1; i < num_samples; ++i)
//     {
//         // 更新距离
//         const auto& current_point = cloud->points[farthest];

//         for (size_t j = 0; j < cloud->points.size(); ++j)
//         {
//             const auto& point = cloud->points[j];
//             float dx = point.x - current_point.x;
//             float dy = point.y - current_point.y;
//             float dz = point.z - current_point.z;
//             float dist = dx * dx + dy * dy + dz * dz;
//             distances[j] = std::min(distances[j], dist);
//         }

//         // 找到最远的点
//         farthest = 0;
//         for (size_t j = 1; j < cloud->points.size(); ++j)
//         {
//             if (distances[j] > distances[farthest])
//             {
//                 farthest = static_cast<int>(j);
//             }
//         }

//         sampled_indices.push_back(farthest);
//     }

//     // 创建采样后的点云
//     PointCloud::Ptr sampled_cloud(new PointCloud);
//     for (int idx : sampled_indices)
//     {
//         sampled_cloud->points.push_back(cloud->points[idx]);
//     }

//     sampled_cloud->width = sampled_cloud->points.size();
//     sampled_cloud->height = 1;
//     sampled_cloud->is_dense = true;

//     return sampled_cloud;
// }

// std::vector<std::array<float, 4>> DSNetInference::forwardInference(
//     const PointCloud::Ptr& preprocessed_cloud)
// {
//     std::vector<std::array<float, 4>> results;
//     results.resize(preprocessed_cloud->points.size());

// #ifdef USE_LIBTORCH
//     try
//     {
//         // 转换为tensor
//         std::vector<float> points_data;
//         points_data.reserve(preprocessed_cloud->points.size() * 3);  // 3坐标

//         for (size_t i = 0; i < preprocessed_cloud->points.size(); ++i)
//         {
//             points_data.push_back(preprocessed_cloud->points[i].x);
//             points_data.push_back(preprocessed_cloud->points[i].y);
//             points_data.push_back(preprocessed_cloud->points[i].z);
//         }

//         auto tensor = torch::from_blob(points_data.data(),
//                                        {1, static_cast<long>(preprocessed_cloud->points.size()),
//                                        3}, torch::kFloat);

//         if (config_.use_gpu && torch::cuda::is_available())
//         {
//             tensor = tensor.to(torch::kCUDA);
//         }

//         // 执行推理
//         std::vector<torch::jit::IValue> inputs;
//         inputs.push_back(tensor);

//         auto output = model_.forward(inputs).toTensor();
//         output = output.to(torch::kCPU);

//         // 转换结果
//         auto output_accessor = output.accessor<float, 3>();
//         for (size_t i = 0; i < preprocessed_cloud->points.size(); ++i)
//         {
//             results[i][0] = output_accessor[0][i][0];  // seal_score
//             results[i][1] = output_accessor[0][i][1];  // wrench_score
//             results[i][2] = output_accessor[0][i][2];  // feasibility_score
//             results[i][3] = output_accessor[0][i][3];  // object_size_score
//         }
//     }
//     catch (const std::exception& e)
//     {
//         std::cerr << "LibTorch推理失败: " << e.what() << std::endl;
//         // 回退到模拟推理
//     }
// #endif

//     // 模拟推理（如果没有LibTorch或推理失败）
//     if (results.empty() || results[0][0] == 0.0f)
//     {
//         std::cout << "使用模拟推理..." << std::endl;

//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::normal_distribution<float> seal_dist(0.0f, 0.5f);
//         std::normal_distribution<float> wrench_dist(-0.4f, 0.2f);
//         std::normal_distribution<float> feasibility_dist(0.5f, 0.3f);
//         std::normal_distribution<float> size_dist(0.5f, 0.2f);

//         for (size_t i = 0; i < preprocessed_cloud->points.size(); ++i)
//         {
//             results[i][0] = seal_dist(gen);         // seal_score
//             results[i][1] = wrench_dist(gen);       // wrench_score
//             results[i][2] = feasibility_dist(gen);  // feasibility_score
//             results[i][3] = size_dist(gen);         // object_size_score
//         }
//     }

//     return results;
// }

// float DSNetInference::computeCompositeScore(const SuctionScores& scores) const
// {
//     return config_.score_weights.seal_weight * scores.seal_score +
//            config_.score_weights.wrench_weight * scores.wrench_score +
//            config_.score_weights.visibility_weight * scores.visibility_score +
//            config_.score_weights.collision_weight * scores.collision_score;
// }

// std::vector<SuctionPoint> DSNetInference::getBestSuctionPoints(
//     const std::vector<SuctionScores>& scores, const PointCloud::Ptr& cloud, int top_k) const
// {
//     // 创建索引数组
//     std::vector<size_t> indices(scores.size());
//     std::iota(indices.begin(), indices.end(), 0);

//     // 按综合评分排序
//     std::partial_sort(indices.begin(),
//                       indices.begin() + std::min(top_k, static_cast<int>(indices.size())),
//                       indices.end(), [&scores](size_t a, size_t b)
//                       { return scores[a].composite_score > scores[b].composite_score; });

//     // 创建最佳吸取点列表
//     std::vector<SuctionPoint> best_points;
//     best_points.reserve(std::min(top_k, static_cast<int>(indices.size())));

//     for (int i = 0; i < std::min(top_k, static_cast<int>(indices.size())); ++i)
//     {
//         size_t idx = indices[i];
//         SuctionPoint point;
//         point.position = cloud->points[idx];
//         point.normal.x = 0.0f;
//         point.normal.y = 0.0f;
//         point.normal.z = 1.0f;  // 默认法向量
//         point.scores = scores[idx];
//         point.index = static_cast<int>(idx);
//         best_points.push_back(point);
//     }

//     return best_points;
// }

// // ==================== Utility Functions ====================

// bool saveInferenceResult(const InferenceResult& result, const std::string& filename)
// {
//     std::ofstream file(filename);
//     if (!file.is_open())
//     {
//         return false;
//     }

//     // 保存基本信息
//     file << "推理时间: " << result.getInferenceTime() << " ms\n";
//     file << "点云大小: " << result.getPointCloudSize() << "\n";
//     file << "最佳点数量: " << result.getBestPointsCount() << "\n\n";

//     // 保存最佳吸取点
//     file << "最佳吸取点:\n";
//     const auto& best_points = result.getBestPoints();
//     for (size_t i = 0; i < best_points.size(); ++i)
//     {
//         const auto& point = best_points[i];
//         file << "排名 " << (i + 1) << ":\n";
//         file << "  位置: (" << point.position.x << ", " << point.position.y << ", "
//              << point.position.z << ")\n";
//         file << "  综合评分: " << point.scores.composite_score << "\n";
//         file << "  密封评分: " << point.scores.seal_score << "\n";
//         file << "  扭矩评分: " << point.scores.wrench_score << "\n";
//         file << "  可见性评分: " << point.scores.visibility_score << "\n";
//         file << "  碰撞评分: " << point.scores.collision_score << "\n\n";
//     }

//     return true;
// }

// }  // namespace dsnet