// #include "dsnet_inference/dsnet_inference_result.h"

// #include <algorithm>
// #include <cmath>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <numeric>
// #include <sstream>

// // 包含 PCL 头文件
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>

// namespace dsnet
// {

// // ==================== 构造函数和析构函数 ====================

// InferenceResult::InferenceResult() : preprocessed_cloud_(nullptr), inference_time_ms_(0.0f) {}

// InferenceResult::InferenceResult(const PointCloud::Ptr& cloud,
//                                  const std::vector<SuctionScores>& scores, float inference_time)
//     : preprocessed_cloud_(cloud), scores_(scores), inference_time_ms_(inference_time)
// {
// }

// InferenceResult::InferenceResult(const InferenceResult& other)
//     : preprocessed_cloud_(other.preprocessed_cloud_),
//       scores_(other.scores_),
//       best_points_(other.best_points_),
//       inference_time_ms_(other.inference_time_ms_)
// {
// }

// InferenceResult& InferenceResult::operator=(const InferenceResult& other)
// {
//     if (this != &other)
//     {
//         preprocessed_cloud_ = other.preprocessed_cloud_;
//         scores_ = other.scores_;
//         best_points_ = other.best_points_;
//         inference_time_ms_ = other.inference_time_ms_;
//     }
//     return *this;
// }

// InferenceResult::InferenceResult(InferenceResult&& other) noexcept
//     : preprocessed_cloud_(std::move(other.preprocessed_cloud_)),
//       scores_(std::move(other.scores_)),
//       best_points_(std::move(other.best_points_)),
//       inference_time_ms_(other.inference_time_ms_)
// {
//     other.inference_time_ms_ = 0.0f;
// }

// InferenceResult& InferenceResult::operator=(InferenceResult&& other) noexcept
// {
//     if (this != &other)
//     {
//         preprocessed_cloud_ = std::move(other.preprocessed_cloud_);
//         scores_ = std::move(other.scores_);
//         best_points_ = std::move(other.best_points_);
//         inference_time_ms_ = other.inference_time_ms_;

//         other.inference_time_ms_ = 0.0f;
//     }
//     return *this;
// }

// // ==================== 基本访问器方法 ====================

// size_t InferenceResult::getPointCloudSize() const
// {
//     return preprocessed_cloud_ ? preprocessed_cloud_->points.size() : 0;
// }

// // ==================== 分析方法 ====================

// float InferenceResult::getMaxCompositeScore() const
// {
//     if (scores_.empty())
//     {
//         return 0.0f;
//     }

//     auto max_it = std::max_element(scores_.begin(), scores_.end(),
//                                    [](const SuctionScores& a, const SuctionScores& b)
//                                    { return a.composite_score < b.composite_score; });

//     return max_it->composite_score;
// }

// float InferenceResult::getAverageCompositeScore() const
// {
//     if (scores_.empty())
//     {
//         return 0.0f;
//     }

//     float sum = std::accumulate(scores_.begin(), scores_.end(), 0.0f,
//                                 [](float total, const SuctionScores& score)
//                                 { return total + score.composite_score; });

//     return sum / static_cast<float>(scores_.size());
// }

// InferenceResult::AllScoreStatistics InferenceResult::computeScoreStatistics() const
// {
//     AllScoreStatistics stats;

//     if (scores_.empty())
//     {
//         return stats;
//     }

//     // 提取各项评分
//     std::vector<float> seal_scores, wrench_scores, visibility_scores, collision_scores,
//         composite_scores;

//     for (const auto& score : scores_)
//     {
//         seal_scores.push_back(score.seal_score);
//         wrench_scores.push_back(score.wrench_score);
//         visibility_scores.push_back(score.visibility_score);
//         collision_scores.push_back(score.collision_score);
//         composite_scores.push_back(score.composite_score);
//     }

//     // 计算各项统计
//     stats.seal_stats = computeSingleScoreStatistics(seal_scores);
//     stats.wrench_stats = computeSingleScoreStatistics(wrench_scores);
//     stats.feasibility_stats = computeSingleScoreStatistics(visibility_scores);
//     stats.object_size_stats = computeSingleScoreStatistics(collision_scores);
//     stats.composite_stats = computeSingleScoreStatistics(composite_scores);

//     return stats;
// }

// InferenceResult::ScoreStatistics InferenceResult::computeSingleScoreStatistics(
//     const std::vector<float>& values) const
// {
//     ScoreStatistics stats;

//     if (values.empty())
//     {
//         return stats;
//     }

//     // 计算最小值和最大值
//     auto minmax = std::minmax_element(values.begin(), values.end());
//     stats.min_score = *minmax.first;
//     stats.max_score = *minmax.second;

//     // 计算平均值
//     float sum = std::accumulate(values.begin(), values.end(), 0.0f);
//     stats.mean_score = sum / static_cast<float>(values.size());

//     // 计算标准差
//     float variance = 0.0f;
//     for (float value : values)
//     {
//         float diff = value - stats.mean_score;
//         variance += diff * diff;
//     }
//     variance /= static_cast<float>(values.size());
//     stats.std_dev = std::sqrt(variance);

//     return stats;
// }

// std::vector<SuctionPoint> InferenceResult::filterPointsByThreshold(float threshold) const
// {
//     std::vector<SuctionPoint> filtered_points;

//     for (const auto& point : best_points_)
//     {
//         if (point.scores.composite_score >= threshold)
//         {
//             filtered_points.push_back(point);
//         }
//     }

//     return filtered_points;
// }

// SuctionPoint InferenceResult::getPointByRank(size_t rank) const
// {
//     if (rank < best_points_.size())
//     {
//         return best_points_[rank];
//     }
//     return SuctionPoint();  // 返回默认构造的空点
// }

// // ==================== 输出方法 ====================

// bool InferenceResult::saveToFile(const std::string& filename, const std::string& format) const
// {
//     if (format == "txt")
//     {
//         return saveToTextFile(filename);
//     }
//     else if (format == "json")
//     {
//         return saveToJsonFile(filename);
//     }
//     else if (format == "csv")
//     {
//         return saveToCsvFile(filename);
//     }
//     else
//     {
//         std::cerr << "不支持的保存格式: " << format << std::endl;
//         return false;
//     }
// }

// bool InferenceResult::saveToTextFile(const std::string& filename) const
// {
//     std::ofstream file(filename);
//     if (!file.is_open())
//     {
//         std::cerr << "无法打开文件进行写入: " << filename << std::endl;
//         return false;
//     }

//     file << "DSNet 推理结果\n";
//     file << "================\n\n";

//     file << "基本信息:\n";
//     file << "  推理时间: " << std::fixed << std::setprecision(3) << inference_time_ms_ << "
//     ms\n"; file << "  点云大小: " << getPointCloudSize() << " 个点\n"; file << "  最佳点数量: "
//     << best_points_.size() << " 个\n\n";

//     // 统计信息
//     auto stats = computeScoreStatistics();
//     file << "评分统计:\n";
//     file << "  综合评分 - 最小值: " << stats.composite_stats.min_score
//          << ", 最大值: " << stats.composite_stats.max_score
//          << ", 平均值: " << stats.composite_stats.mean_score
//          << ", 标准差: " << stats.composite_stats.std_dev << "\n";
//     file << "  密封评分 - 最小值: " << stats.seal_stats.min_score
//          << ", 最大值: " << stats.seal_stats.max_score
//          << ", 平均值: " << stats.seal_stats.mean_score << ", 标准差: " <<
//          stats.seal_stats.std_dev
//          << "\n";
//     file << "  扭矩评分 - 最小值: " << stats.wrench_stats.min_score
//          << ", 最大值: " << stats.wrench_stats.max_score
//          << ", 平均值: " << stats.wrench_stats.mean_score
//          << ", 标准差: " << stats.wrench_stats.std_dev << "\n\n";

//     // 最佳吸取点
//     file << "最佳吸取点:\n";
//     for (size_t i = 0; i < best_points_.size(); ++i)
//     {
//         const auto& point = best_points_[i];
//         file << "  排名 " << (i + 1) << ":\n";
//         file << "    位置: (" << point.position.x << ", " << point.position.y << ", "
//              << point.position.z << ")\n";
//         file << "    法向量: (" << point.normal.x << ", " << point.normal.y << ", "
//              << point.normal.z << ")\n";
//         file << "    索引: " << point.index << "\n";
//         file << "    综合评分: " << point.scores.composite_score << "\n";
//         file << "    密封评分: " << point.scores.seal_score << "\n";
//         file << "    扭矩评分: " << point.scores.wrench_score << "\n";
//         file << "    可见性评分: " << point.scores.visibility_score << "\n";
//         file << "    碰撞评分: " << point.scores.collision_score << "\n\n";
//     }

//     return true;
// }

// bool InferenceResult::saveToJsonFile(const std::string& filename) const
// {
//     std::ofstream file(filename);
//     if (!file.is_open())
//     {
//         return false;
//     }

//     file << "{\n";
//     file << "  \"inference_time_ms\": " << inference_time_ms_ << ",\n";
//     file << "  \"point_cloud_size\": " << getPointCloudSize() << ",\n";
//     file << "  \"best_points_count\": " << best_points_.size() << ",\n";

//     auto stats = computeScoreStatistics();
//     file << "  \"statistics\": {\n";
//     file << "    \"composite_score\": {\n";
//     file << "      \"min\": " << stats.composite_stats.min_score << ",\n";
//     file << "      \"max\": " << stats.composite_stats.max_score << ",\n";
//     file << "      \"mean\": " << stats.composite_stats.mean_score << ",\n";
//     file << "      \"std_dev\": " << stats.composite_stats.std_dev << "\n";
//     file << "    }\n";
//     file << "  },\n";

//     file << "  \"best_points\": [\n";
//     for (size_t i = 0; i < best_points_.size(); ++i)
//     {
//         const auto& point = best_points_[i];
//         file << "    {\n";
//         file << "      \"rank\": " << (i + 1) << ",\n";
//         file << "      \"position\": [" << point.position.x << ", " << point.position.y << ", "
//              << point.position.z << "],\n";
//         file << "      \"normal\": [" << point.normal.x << ", " << point.normal.y << ", "
//              << point.normal.z << "],\n";
//         file << "      \"index\": " << point.index << ",\n";
//         file << "      \"scores\": {\n";
//         file << "        \"composite\": " << point.scores.composite_score << ",\n";
//         file << "        \"seal\": " << point.scores.seal_score << ",\n";
//         file << "        \"wrench\": " << point.scores.wrench_score << ",\n";
//         file << "        \"visibility\": " << point.scores.visibility_score << ",\n";
//         file << "        \"collision\": " << point.scores.collision_score << "\n";
//         file << "      }\n";
//         file << "    }";
//         if (i < best_points_.size() - 1)
//             file << ",";
//         file << "\n";
//     }
//     file << "  ]\n";
//     file << "}\n";

//     return true;
// }

// bool InferenceResult::saveToCsvFile(const std::string& filename) const
// {
//     std::ofstream file(filename);
//     if (!file.is_open())
//     {
//         return false;
//     }

//     // CSV 头部
//     file << "rank,position_x,position_y,position_z,normal_x,normal_y,normal_z,index,";
//     file << "composite_score,seal_score,wrench_score,visibility_score,collision_score\n";

//     // 数据行
//     for (size_t i = 0; i < best_points_.size(); ++i)
//     {
//         const auto& point = best_points_[i];
//         file << (i + 1) << ",";
//         file << point.position.x << "," << point.position.y << "," << point.position.z << ",";
//         file << point.normal.x << "," << point.normal.y << "," << point.normal.z << ",";
//         file << point.index << ",";
//         file << point.scores.composite_score << ",";
//         file << point.scores.seal_score << ",";
//         file << point.scores.wrench_score << ",";
//         file << point.scores.visibility_score << ",";
//         file << point.scores.collision_score << "\n";
//     }

//     return true;
// }

// std::string InferenceResult::getSummaryString() const
// {
//     std::ostringstream oss;

//     oss << "DSNet 推理结果摘要:\n";
//     oss << "  推理时间: " << std::fixed << std::setprecision(3) << inference_time_ms_ << " ms\n";
//     oss << "  点云大小: " << getPointCloudSize() << " 个点\n";
//     oss << "  最佳点数量: " << best_points_.size() << " 个\n";

//     if (!scores_.empty())
//     {
//         oss << "  最高综合评分: " << getMaxCompositeScore() << "\n";
//         oss << "  平均综合评分: " << getAverageCompositeScore() << "\n";
//     }

//     if (!best_points_.empty())
//     {
//         const auto& best_point = best_points_[0];
//         oss << "  最佳吸取点位置: (" << best_point.position.x << ", " << best_point.position.y
//             << ", " << best_point.position.z << ")\n";
//         oss << "  最佳吸取点评分: " << best_point.scores.composite_score << "\n";
//     }

//     return oss.str();
// }

// void InferenceResult::printSummary() const
// {
//     std::cout << getSummaryString();
// }

// // ==================== 验证方法 ====================

// bool InferenceResult::isValid() const
// {
//     // 检查基本数据有效性
//     if (!preprocessed_cloud_ || preprocessed_cloud_->points.empty())
//     {
//         return false;
//     }

//     if (scores_.empty())
//     {
//         return false;
//     }

//     // 检查点云和评分数量是否匹配
//     if (preprocessed_cloud_->points.size() != scores_.size())
//     {
//         return false;
//     }

//     // 检查推理时间是否合理
//     if (inference_time_ms_ < 0.0f)
//     {
//         return false;
//     }

//     return true;
// }

// void InferenceResult::clear()
// {
//     preprocessed_cloud_.reset();
//     scores_.clear();
//     best_points_.clear();
//     inference_time_ms_ = 0.0f;
// }

// }  // namespace dsnet