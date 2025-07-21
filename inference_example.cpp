#include <chrono>
#include <iostream>
#include <memory>

#include "dsnet_inference/dsnet_inference.h"

using namespace dsnet;

int main(int argc, char* argv[])
{
    // 1. 配置推理参数
    InferenceConfig config;
    config.num_points = 16384;
    config.diffusion_steps = 50;
    config.use_gpu = true;
    config.device = "cuda";

    // 设置评分权重
    config.score_weights.seal_weight = 0.3f;
    config.score_weights.wrench_weight = 0.3f;
    config.score_weights.visibility_weight = 0.2f;
    config.score_weights.collision_weight = 0.2f;

    // 2. 创建推理器实例
    std::string model_path = "./Data/model/dsnet_model.pt";  // TorchScript模型路径
    DSNetInference inference(model_path, config);

    // 3. 初始化推理器
    if (!inference.initialize())
    {
        std::cerr << "推理器初始化失败!" << std::endl;
        return -1;
    }

    // 4. 加载测试点云
    std::string cloud_path = "./Data/test_cloud.pcd";
    auto input_cloud = loadPointCloudFromFile(cloud_path);
    if (!input_cloud || input_cloud->empty())
    {
        std::cerr << "无法加载点云文件: " << cloud_path << std::endl;
        return -1;
    }

    std::cout << "加载点云成功，包含 " << input_cloud->size() << " 个点" << std::endl;

    // 5. 执行推理
    std::cout << "\n开始推理..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    InferenceResult result = inference.predict(input_cloud);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "推理完成，耗时: " << duration.count() << " ms" << std::endl;

    // 6. 分析结果
    std::cout << "\n推理结果统计:" << std::endl;
    std::cout << "输入点数: " << result.getInputSize() << std::endl;
    std::cout << "输出点数: " << result.getOutputSize() << std::endl;
    std::cout << "推理时间: " << result.getInferenceTime() << " ms" << std::endl;

    auto stats = result.getStatistics();
    std::cout << "\n评分统计:" << std::endl;
    std::cout << "平均密封评分: " << stats.mean_seal_score << " ± " << stats.std_seal_score
              << std::endl;
    std::cout << "平均扭矩评分: " << stats.mean_wrench_score << " ± " << stats.std_wrench_score
              << std::endl;
    std::cout << "平均可行性评分: " << stats.mean_feasibility_score << " ± "
              << stats.std_feasibility_score << std::endl;
    std::cout << "平均综合评分: " << stats.mean_composite_score << " ± "
              << stats.std_composite_score << std::endl;

    // 7. 获取最佳吸取点
    auto suction_points = result.getSuctionPoints();
    auto best_points = inference.getBestSuctionPoints(
        [&suction_points]()
        {
            std::vector<SuctionScores> scores;
            for (const auto& point : suction_points)
            {
                scores.push_back(point.scores);
            }
            return scores;
        }(),
        input_cloud,
        10  // 获取前10个最佳点
    );

    std::cout << "\n前10个最佳吸取点:" << std::endl;
    for (size_t i = 0; i < best_points.size(); ++i)
    {
        const auto& point = best_points[i];
        std::cout << "点 " << (i + 1) << ": "
                  << "位置(" << point.position.x << ", " << point.position.y << ", "
                  << point.position.z << ") "
                  << "综合评分: " << point.scores.composite_score << std::endl;
    }

    // 8. 保存结果
    std::string result_dir = "./results/";

    // 保存为不同格式
    result.saveToTXT(result_dir + "inference_result.txt");
    result.saveToJSON(result_dir + "inference_result.json");
    result.saveToCSV(result_dir + "inference_result.csv");

    std::cout << "\n结果已保存到: " << result_dir << std::endl;

    // 9. 可选：可视化结果（需要OpenCV支持）
#ifdef USE_OPENCV
    if (visualizeResult(result, result_dir + "visualization.png"))
    {
        std::cout << "可视化结果已保存到: " << result_dir << "visualization.png" << std::endl;
    }
#endif

    return 0;
}
