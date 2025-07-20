#include "dsnet_inference.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

namespace dsnet {

// ==================== PointCloud Implementation ====================

bool PointCloud::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }
    
    clear();
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x, y, z, nx = 0, ny = 0, nz = 0;
        
        if (!(iss >> x >> y >> z)) {
            continue; // 跳过无效行
        }
        
        // 尝试读取法向量
        iss >> nx >> ny >> nz;
        bool has_normal = !iss.fail();
        
        points.emplace_back(x, y, z);
        if (has_normal) {
            normals.emplace_back(nx, ny, nz);
        }
    }
    
    // 如果只有部分点有法向量，清空法向量数组
    if (!normals.empty() && normals.size() != points.size()) {
        normals.clear();
    }
    
    std::cout << "加载点云: " << points.size() << " 个点";
    if (hasNormals()) {
        std::cout << " (包含法向量)";
    }
    std::cout << std::endl;
    
    return !points.empty();
}

bool PointCloud::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < points.size(); ++i) {
        file << points[i].x << " " << points[i].y << " " << points[i].z;
        
        if (hasNormals()) {
            file << " " << normals[i].x << " " << normals[i].y << " " << normals[i].z;
        }
        
        file << "\n";
    }
    
    return true;
}

void PointCloud::normalize() {
    if (points.empty()) return;
    
    // 计算质心
    Eigen::Vector3f centroid(0, 0, 0);
    for (const auto& point : points) {
        centroid += point.toEigen();
    }
    centroid /= static_cast<float>(points.size());
    
    // 移动到原点
    for (auto& point : points) {
        Eigen::Vector3f vec = point.toEigen() - centroid;
        point = Point3D::fromEigen(vec);
    }
    
    // 计算最大距离
    float max_distance = 0.0f;
    for (const auto& point : points) {
        float distance = point.toEigen().norm();
        max_distance = std::max(max_distance, distance);
    }
    
    // 缩放到单位球
    if (max_distance > 1e-8f) {
        for (auto& point : points) {
            Eigen::Vector3f vec = point.toEigen() / max_distance;
            point = Point3D::fromEigen(vec);
        }
    }
}

std::pair<Point3D, Point3D> PointCloud::getBoundingBox() const {
    if (points.empty()) {
        return {Point3D(), Point3D()};
    }
    
    Point3D min_pt = points[0];
    Point3D max_pt = points[0];
    
    for (const auto& point : points) {
        min_pt.x = std::min(min_pt.x, point.x);
        min_pt.y = std::min(min_pt.y, point.y);
        min_pt.z = std::min(min_pt.z, point.z);
        
        max_pt.x = std::max(max_pt.x, point.x);
        max_pt.y = std::max(max_pt.y, point.y);
        max_pt.z = std::max(max_pt.z, point.z);
    }
    
    return {min_pt, max_pt};
}

// ==================== DSNetInference Implementation ====================

DSNetInference::DSNetInference(const std::string& model_path, const InferenceConfig& config)
    : model_path_(model_path), config_(config), initialized_(false) {
}

DSNetInference::~DSNetInference() {
    // 清理资源
}

bool DSNetInference::initialize() {
    std::cout << "初始化 DSNet 推理器..." << std::endl;
    std::cout << "模型路径: " << model_path_ << std::endl;
    std::cout << "配置参数:" << std::endl;
    std::cout << "  - 采样点数: " << config_.num_points << std::endl;
    std::cout << "  - 扩散步数: " << config_.diffusion_steps << std::endl;
    std::cout << "  - 设备: " << config_.device << std::endl;
    
    // 检查模型文件是否存在
    std::ifstream model_file(model_path_);
    if (!model_file.good()) {
        std::cerr << "错误: 模型文件不存在: " << model_path_ << std::endl;
        return false;
    }
    
    // 加载模型
    if (!loadModel()) {
        std::cerr << "错误: 模型加载失败" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "DSNet 推理器初始化成功!" << std::endl;
    return true;
}

bool DSNetInference::loadModel() {
#ifdef USE_LIBTORCH
    try {
        // 使用LibTorch加载模型
        model_ = torch::jit::load(model_path_);
        model_.eval();
        
        if (config_.use_gpu && torch::cuda::is_available()) {
            model_.to(torch::kCUDA);
            std::cout << "模型已加载到GPU" << std::endl;
        } else {
            model_.to(torch::kCPU);
            std::cout << "模型已加载到CPU" << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "LibTorch模型加载失败: " << e.what() << std::endl;
        return false;
    }
#else
    // 没有LibTorch，使用占位符实现
    std::cout << "警告: 未启用LibTorch，使用模拟推理" << std::endl;
    return true;
#endif
}

InferenceResult DSNetInference::predict(const PointCloud& input_cloud) {
    if (!initialized_) {
        throw std::runtime_error("推理器未初始化");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    InferenceResult result;
    
    // 预处理点云
    result.preprocessed_cloud = preprocessPointCloud(input_cloud);
    
    // 执行推理
    auto raw_scores = forwardInference(result.preprocessed_cloud);
    
    // 转换为SuctionScores格式
    result.scores.resize(raw_scores.size());
    for (size_t i = 0; i < raw_scores.size(); ++i) {
        result.scores[i].seal_score = raw_scores[i][0];
        result.scores[i].wrench_score = raw_scores[i][1];
        result.scores[i].feasibility_score = raw_scores[i][2];
        result.scores[i].object_size_score = raw_scores[i][3];
        result.scores[i].composite_score = computeCompositeScore(result.scores[i]);
    }
    
    // 获取最佳吸取点
    result.best_points = getBestSuctionPoints(result.scores, result.preprocessed_cloud, 10);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.inference_time_ms = static_cast<float>(duration.count());
    
    std::cout << "推理完成，用时: " << result.inference_time_ms << " ms" << std::endl;
    
    return result;
}

std::vector<InferenceResult> DSNetInference::predictBatch(const std::vector<PointCloud>& input_clouds) {
    std::vector<InferenceResult> results;
    results.reserve(input_clouds.size());
    
    for (const auto& cloud : input_clouds) {
        results.push_back(predict(cloud));
    }
    
    return results;
}

PointCloud DSNetInference::preprocessPointCloud(const PointCloud& input_cloud) {
    std::cout << "预处理点云: " << input_cloud.size() << " 个点" << std::endl;
    
    PointCloud processed_cloud = input_cloud;
    
    // 归一化
    processed_cloud.normalize();
    
    // 采样到目标点数
    if (static_cast<int>(processed_cloud.size()) != config_.num_points) {
        if (static_cast<int>(processed_cloud.size()) > config_.num_points) {
            // 下采样
            processed_cloud = fpsSubsample(processed_cloud, config_.num_points);
        } else {
            // 上采样（重复采样）
            PointCloud upsampled_cloud;
            int repeat_times = (config_.num_points + static_cast<int>(processed_cloud.size()) - 1) / 
                              static_cast<int>(processed_cloud.size());
            
            for (int r = 0; r < repeat_times && static_cast<int>(upsampled_cloud.size()) < config_.num_points; ++r) {
                for (size_t i = 0; i < processed_cloud.size() && static_cast<int>(upsampled_cloud.size()) < config_.num_points; ++i) {
                    upsampled_cloud.addPoint(processed_cloud.points[i], 
                                           processed_cloud.hasNormals() ? processed_cloud.normals[i] : Point3D());
                }
            }
            processed_cloud = upsampled_cloud;
        }
    }
    
    // 如果没有法向量，添加零向量
    if (!processed_cloud.hasNormals()) {
        processed_cloud.normals.resize(processed_cloud.size(), Point3D(0, 0, 0));
        std::cout << "警告: 未提供法向量，使用零向量" << std::endl;
    }
    
    std::cout << "预处理后点云: " << processed_cloud.size() << " 个点" << std::endl;
    
    return processed_cloud;
}

PointCloud DSNetInference::fpsSubsample(const PointCloud& cloud, int num_samples) {
    if (static_cast<int>(cloud.size()) <= num_samples) {
        return cloud;
    }
    
    std::vector<int> sampled_indices;
    std::vector<float> distances(cloud.size(), std::numeric_limits<float>::max());
    
    // 随机选择第一个点
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(cloud.size()) - 1);
    
    int farthest = dis(gen);
    sampled_indices.push_back(farthest);
    
    for (int i = 1; i < num_samples; ++i) {
        // 更新距离
        Eigen::Vector3f current_point = cloud.points[farthest].toEigen();
        
        for (size_t j = 0; j < cloud.size(); ++j) {
            Eigen::Vector3f point = cloud.points[j].toEigen();
            float dist = (point - current_point).squaredNorm();
            distances[j] = std::min(distances[j], dist);
        }
        
        // 找到最远的点
        farthest = 0;
        for (size_t j = 1; j < cloud.size(); ++j) {
            if (distances[j] > distances[farthest]) {
                farthest = static_cast<int>(j);
            }
        }
        
        sampled_indices.push_back(farthest);
    }
    
    // 创建采样后的点云
    PointCloud sampled_cloud;
    for (int idx : sampled_indices) {
        sampled_cloud.addPoint(cloud.points[idx], 
                              cloud.hasNormals() ? cloud.normals[idx] : Point3D());
    }
    
    return sampled_cloud;
}

std::vector<std::array<float, 4>> DSNetInference::forwardInference(const PointCloud& preprocessed_cloud) {
    std::vector<std::array<float, 4>> results;
    results.resize(preprocessed_cloud.size());
    
#ifdef USE_LIBTORCH
    try {
        // 转换为tensor
        std::vector<float> points_data;
        points_data.reserve(preprocessed_cloud.size() * 6); // 3坐标 + 3法向量
        
        for (size_t i = 0; i < preprocessed_cloud.size(); ++i) {
            points_data.push_back(preprocessed_cloud.points[i].x);
            points_data.push_back(preprocessed_cloud.points[i].y);
            points_data.push_back(preprocessed_cloud.points[i].z);
            points_data.push_back(preprocessed_cloud.normals[i].x);
            points_data.push_back(preprocessed_cloud.normals[i].y);
            points_data.push_back(preprocessed_cloud.normals[i].z);
        }
        
        auto tensor = torch::from_blob(points_data.data(), 
                                     {1, static_cast<long>(preprocessed_cloud.size()), 6}, 
                                     torch::kFloat);
        
        if (config_.use_gpu && torch::cuda::is_available()) {
            tensor = tensor.to(torch::kCUDA);
        }
        
        // 执行推理
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        auto output = model_.forward(inputs).toTensor();
        output = output.to(torch::kCPU);
        
        // 转换结果
        auto output_accessor = output.accessor<float, 3>();
        for (size_t i = 0; i < preprocessed_cloud.size(); ++i) {
            results[i][0] = output_accessor[0][i][0]; // seal_score
            results[i][1] = output_accessor[0][i][1]; // wrench_score
            results[i][2] = output_accessor[0][i][2]; // feasibility_score
            results[i][3] = output_accessor[0][i][3]; // object_size_score
        }
        
    } catch (const std::exception& e) {
        std::cerr << "LibTorch推理失败: " << e.what() << std::endl;
        // 回退到模拟推理
    }
#endif
    
    // 模拟推理（如果没有LibTorch或推理失败）
    if (results.empty() || results[0][0] == 0.0f) {
        std::cout << "使用模拟推理..." << std::endl;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> seal_dist(0.0f, 0.5f);
        std::normal_distribution<float> wrench_dist(-0.4f, 0.2f);
        std::normal_distribution<float> feasibility_dist(0.5f, 0.3f);
        std::normal_distribution<float> size_dist(0.5f, 0.2f);
        
        for (size_t i = 0; i < preprocessed_cloud.size(); ++i) {
            results[i][0] = seal_dist(gen);        // seal_score
            results[i][1] = wrench_dist(gen);      // wrench_score
            results[i][2] = feasibility_dist(gen); // feasibility_score
            results[i][3] = size_dist(gen);        // object_size_score
        }
    }
    
    return results;
}

float DSNetInference::computeCompositeScore(const SuctionScores& scores) const {
    return config_.score_weights.seal_weight * scores.seal_score +
           config_.score_weights.wrench_weight * scores.wrench_score +
           config_.score_weights.feasibility_weight * scores.feasibility_score +
           config_.score_weights.object_size_weight * scores.object_size_score;
}

std::vector<SuctionPoint> DSNetInference::getBestSuctionPoints(
    const std::vector<SuctionScores>& scores,
    const PointCloud& cloud,
    int top_k) const {
    
    // 创建索引数组
    std::vector<size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // 按综合评分排序
    std::partial_sort(indices.begin(), indices.begin() + std::min(top_k, static_cast<int>(indices.size())), indices.end(),
                     [&scores](size_t a, size_t b) {
                         return scores[a].composite_score > scores[b].composite_score;
                     });
    
    // 创建最佳吸取点列表
    std::vector<SuctionPoint> best_points;
    best_points.reserve(std::min(top_k, static_cast<int>(indices.size())));
    
    for (int i = 0; i < std::min(top_k, static_cast<int>(indices.size())); ++i) {
        size_t idx = indices[i];
        SuctionPoint point;
        point.position = cloud.points[idx];
        point.normal = cloud.hasNormals() ? cloud.normals[idx] : Point3D(0, 0, 1);
        point.scores = scores[idx];
        point.index = static_cast<int>(idx);
        best_points.push_back(point);
    }
    
    return best_points;
}

// ==================== Utility Functions ====================

PointCloud loadPointCloudFromFile(const std::string& filename) {
    PointCloud cloud;
    cloud.loadFromFile(filename);
    return cloud;
}

bool savePointCloudToFile(const PointCloud& cloud, const std::string& filename) {
    return cloud.saveToFile(filename);
}

bool saveInferenceResult(const InferenceResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // 保存基本信息
    file << "推理时间: " << result.inference_time_ms << " ms\n";
    file << "点云大小: " << result.preprocessed_cloud.size() << "\n";
    file << "最佳点数量: " << result.best_points.size() << "\n\n";
    
    // 保存最佳吸取点
    file << "最佳吸取点:\n";
    for (size_t i = 0; i < result.best_points.size(); ++i) {
        const auto& point = result.best_points[i];
        file << "排名 " << (i + 1) << ":\n";
        file << "  位置: (" << point.position.x << ", " << point.position.y << ", " << point.position.z << ")\n";
        file << "  综合评分: " << point.scores.composite_score << "\n";
        file << "  密封评分: " << point.scores.seal_score << "\n";
        file << "  扭矩评分: " << point.scores.wrench_score << "\n";
        file << "  可行性评分: " << point.scores.feasibility_score << "\n";
        file << "  尺寸评分: " << point.scores.object_size_score << "\n\n";
    }
    
    return true;
}

} // namespace dsnet