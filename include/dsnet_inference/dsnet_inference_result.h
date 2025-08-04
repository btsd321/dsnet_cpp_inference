// #pragma once

// #include <chrono>
// #include <memory>
// #include <string>
// #include <vector>

// #include "dsnet_inference/dsnet_utils.h"

// namespace dsnet
// {

// // 使用 PCL 类型的别名
// typedef pcl::PointXYZ Point3D;
// typedef pcl::PointCloud<Point3D> PointCloud;

// /**
//  * @brief 吸取评分结构
//  */
// struct SuctionScores
// {
//     float seal_score;        // 密封评分
//     float wrench_score;      // 扭矩评分
//     float visibility_score;  // 可见性评分
//     float collision_score;   // 碰撞评分
//     float composite_score;   // 综合评分

//     SuctionScores()
//         : seal_score(0.0f),
//           wrench_score(0.0f),
//           visibility_score(0.0f),
//           collision_score(0.0f),
//           composite_score(0.0f)
//     {
//     }
// };

// /**
//  * @brief 吸取点结构
//  */
// struct SuctionPoint
// {
//     Point3D position;      // 3D位置
//     Point3D normal;        // 法向量
//     SuctionScores scores;  // 评分
//     int index;             // 在点云中的索引

//     SuctionPoint() : index(-1)
//     {
//         position.x = position.y = position.z = 0.0f;
//         normal.x = normal.y = normal.z = 0.0f;
//     }
// };

// /**
//  * @brief 推理结果类
//  * 包含完整的推理结果信息，包括预处理后的点云、各种评分、最佳吸取点等
//  */
// class InferenceResult
// {
//    public:
//     /**
//      * @brief 默认构造函数
//      */
//     InferenceResult();

//     /**
//      * @brief 构造函数
//      * @param cloud 预处理后的点云
//      * @param scores 吸取评分数组
//      * @param inference_time 推理时间（毫秒）
//      */
//     InferenceResult(const PointCloud::Ptr& cloud, const std::vector<SuctionScores>& scores,
//                     float inference_time = 0.0f);

//     /**
//      * @brief 拷贝构造函数
//      */
//     InferenceResult(const InferenceResult& other);

//     /**
//      * @brief 赋值运算符
//      */
//     InferenceResult& operator=(const InferenceResult& other);

//     /**
//      * @brief 移动构造函数
//      */
//     InferenceResult(InferenceResult&& other) noexcept;

//     /**
//      * @brief 移动赋值运算符
//      */
//     InferenceResult& operator=(InferenceResult&& other) noexcept;

//     /**
//      * @brief 析构函数
//      */
//     ~InferenceResult() = default;

//     // ==================== 访问器方法 ====================

//     /**
//      * @brief 获取预处理后的点云
//      * @return 点云指针
//      */
//     PointCloud::Ptr getPreprocessedCloud() const
//     {
//         return preprocessed_cloud_;
//     }

//     /**
//      * @brief 设置预处理后的点云
//      * @param cloud 点云指针
//      */
//     void setPreprocessedCloud(const PointCloud::Ptr& cloud)
//     {
//         preprocessed_cloud_ = cloud;
//     }

//     /**
//      * @brief 获取吸取评分数组
//      * @return 评分数组
//      */
//     const std::vector<SuctionScores>& getScores() const
//     {
//         return scores_;
//     }

//     /**
//      * @brief 设置吸取评分数组
//      * @param scores 评分数组
//      */
//     void setScores(const std::vector<SuctionScores>& scores)
//     {
//         scores_ = scores;
//     }

//     /**
//      * @brief 获取最佳吸取点列表
//      * @return 最佳吸取点列表
//      */
//     const std::vector<SuctionPoint>& getBestPoints() const
//     {
//         return best_points_;
//     }

//     /**
//      * @brief 设置最佳吸取点列表
//      * @param points 最佳吸取点列表
//      */
//     void setBestPoints(const std::vector<SuctionPoint>& points)
//     {
//         best_points_ = points;
//     }

//     /**
//      * @brief 获取推理时间（毫秒）
//      * @return 推理时间
//      */
//     float getInferenceTime() const
//     {
//         return inference_time_ms_;
//     }

//     /**
//      * @brief 设置推理时间（毫秒）
//      * @param time 推理时间
//      */
//     void setInferenceTime(float time)
//     {
//         inference_time_ms_ = time;
//     }

//     /**
//      * @brief 获取点云大小
//      * @return 点云中的点数
//      */
//     size_t getPointCloudSize() const;

//     /**
//      * @brief 获取最佳点数量
//      * @return 最佳吸取点数量
//      */
//     size_t getBestPointsCount() const
//     {
//         return best_points_.size();
//     }

//     // ==================== 分析方法 ====================

//     /**
//      * @brief 获取最高综合评分
//      * @return 最高综合评分值
//      */
//     float getMaxCompositeScore() const;

//     /**
//      * @brief 获取平均综合评分
//      * @return 平均综合评分值
//      */
//     float getAverageCompositeScore() const;

//     /**
//      * @brief 获取评分统计信息
//      * @return 包含最小值、最大值、平均值、标准差的统计信息
//      */
//     struct ScoreStatistics
//     {
//         float min_score;
//         float max_score;
//         float mean_score;
//         float std_dev;

//         ScoreStatistics() : min_score(0.0f), max_score(0.0f), mean_score(0.0f), std_dev(0.0f) {}
//     };

//     /**
//      * @brief 计算各项评分的统计信息
//      * @return 各项评分的统计信息
//      */
//     struct AllScoreStatistics
//     {
//         ScoreStatistics seal_stats;
//         ScoreStatistics wrench_stats;
//         ScoreStatistics feasibility_stats;
//         ScoreStatistics object_size_stats;
//         ScoreStatistics composite_stats;
//     };

//     AllScoreStatistics computeScoreStatistics() const;

//     /**
//      * @brief 根据阈值过滤最佳点
//      * @param threshold 评分阈值
//      * @return 过滤后的吸取点列表
//      */
//     std::vector<SuctionPoint> filterPointsByThreshold(float threshold) const;

//     /**
//      * @brief 获取指定排名的吸取点
//      * @param rank 排名（从0开始）
//      * @return 吸取点，如果排名无效则返回空的SuctionPoint
//      */
//     SuctionPoint getPointByRank(size_t rank) const;

//     // ==================== 输出方法 ====================

//     /**
//      * @brief 将结果保存到文件
//      * @param filename 文件名
//      * @param format 保存格式 ("txt", "json", "csv")
//      * @return 是否保存成功
//      */
//     bool saveToFile(const std::string& filename, const std::string& format = "txt") const;

//     /**
//      * @brief 获取结果摘要字符串
//      * @return 结果摘要
//      */
//     std::string getSummaryString() const;

//     /**
//      * @brief 打印结果摘要到控制台
//      */
//     void printSummary() const;

//     // ==================== 验证方法 ====================

//     /**
//      * @brief 检查结果是否有效
//      * @return 是否有效
//      */
//     bool isValid() const;

//     /**
//      * @brief 清空所有数据
//      */
//     void clear();

//    private:
//     // 成员变量
//     PointCloud::Ptr preprocessed_cloud_;
//     std::vector<SuctionScores> scores_;
//     std::vector<SuctionPoint> best_points_;
//     float inference_time_ms_;

//     /**
//      * @brief 计算单项评分的统计信息
//      * @param values 评分值数组
//      * @return 统计信息
//      */
//     ScoreStatistics computeSingleScoreStatistics(const std::vector<float>& values) const;

//     /**
//      * @brief 保存结果到文本文件
//      * @param filename 文件名
//      * @return 是否成功
//      */
//     bool saveToTextFile(const std::string& filename) const;

//     /**
//      * @brief 保存结果到JSON文件
//      * @param filename 文件名
//      * @return 是否成功
//      */
//     bool saveToJsonFile(const std::string& filename) const;

//     /**
//      * @brief 保存结果到CSV文件
//      * @param filename 文件名
//      * @return 是否成功
//      */
//     bool saveToCsvFile(const std::string& filename) const;
// };

// }  // namespace dsnet
