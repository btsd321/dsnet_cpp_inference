#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <iostream>

#include "dsnet_inference/dsnet_inference_result.h"

int main()
{
    std::cout << "测试 InferenceResult 类..." << std::endl;

    // 创建测试点云
    dsnet::PointCloud::Ptr test_cloud(new dsnet::PointCloud);
    test_cloud->points.resize(100);

    for (size_t i = 0; i < test_cloud->points.size(); ++i)
    {
        test_cloud->points[i].x = static_cast<float>(i) * 0.01f;
        test_cloud->points[i].y = static_cast<float>(i) * 0.01f;
        test_cloud->points[i].z = static_cast<float>(i) * 0.01f;
    }

    test_cloud->width = test_cloud->points.size();
    test_cloud->height = 1;
    test_cloud->is_dense = true;

    // 创建测试评分
    std::vector<dsnet::SuctionScores> test_scores(100);
    for (size_t i = 0; i < test_scores.size(); ++i)
    {
        test_scores[i].seal_score = static_cast<float>(i) / 100.0f;
        test_scores[i].wrench_score = static_cast<float>(i) / 100.0f * 0.8f;
        test_scores[i].visibility_score = static_cast<float>(i) / 100.0f * 0.9f;
        test_scores[i].collision_score = static_cast<float>(i) / 100.0f * 0.7f;
        test_scores[i].composite_score =
            (test_scores[i].seal_score + test_scores[i].wrench_score +
             test_scores[i].visibility_score + test_scores[i].collision_score) /
            4.0f;
    }

    // 测试 InferenceResult 构造函数
    dsnet::InferenceResult result(test_cloud, test_scores, 150.5f);

    // 测试基本功能
    std::cout << "点云大小: " << result.getPointCloudSize() << std::endl;
    std::cout << "推理时间: " << result.getInferenceTime() << " ms" << std::endl;
    std::cout << "最高评分: " << result.getMaxCompositeScore() << std::endl;
    std::cout << "平均评分: " << result.getAverageCompositeScore() << std::endl;

    // 创建一些最佳点
    std::vector<dsnet::SuctionPoint> best_points;
    for (int i = 95; i < 100; ++i)
    {
        dsnet::SuctionPoint point;
        point.position = test_cloud->points[i];
        point.normal.x = 0.0f;
        point.normal.y = 0.0f;
        point.normal.z = 1.0f;
        point.scores = test_scores[i];
        point.index = i;
        best_points.push_back(point);
    }
    result.setBestPoints(best_points);

    // 测试统计功能
    auto stats = result.computeScoreStatistics();
    std::cout << "综合评分统计:" << std::endl;
    std::cout << "  最小值: " << stats.composite_stats.min_score << std::endl;
    std::cout << "  最大值: " << stats.composite_stats.max_score << std::endl;
    std::cout << "  平均值: " << stats.composite_stats.mean_score << std::endl;
    std::cout << "  标准差: " << stats.composite_stats.std_dev << std::endl;

    // 测试摘要功能
    std::cout << "\n" << result.getSummaryString() << std::endl;

    // 测试保存功能
    bool save_result = result.saveToFile("test_result.txt", "txt");
    std::cout << "保存到文件: " << (save_result ? "成功" : "失败") << std::endl;

    // 测试验证功能
    std::cout << "结果有效性: " << (result.isValid() ? "有效" : "无效") << std::endl;

    std::cout << "测试完成!" << std::endl;
    return 0;
}
