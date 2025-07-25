#include <iostream>
#include <memory>
#include <vector>

#include "dsnet_inference/dsnet_inference.h"

using namespace dsnet;

// 创建测试点云
PointCloud::Ptr createTestPointCloud()
{
    auto cloud = std::make_shared<PointCloud>();

    // 创建一个简单的立方体点云
    for (float x = -0.5f; x <= 0.5f; x += 0.1f)
    {
        for (float y = -0.5f; y <= 0.5f; y += 0.1f)
        {
            for (float z = -0.5f; z <= 0.5f; z += 0.1f)
            {
                Point3D point;
                point.x = x;
                point.y = y;
                point.z = z;
                cloud->points.push_back(point);
            }
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    std::cout << "创建测试点云，包含 " << cloud->size() << " 个点" << std::endl;
    return cloud;
}

// 测试推理结果类
bool testInferenceResult()
{
    std::cout << "\n=== 测试推理结果类 ===" << std::endl;

    // 创建测试点云
    auto cloud = createTestPointCloud();

    // 创建推理结果
    InferenceResult result;
    result.setInputCloud(cloud);

    // 创建模拟的吸取点和评分
    std::vector<SuctionPoint> suction_points;
    for (size_t i = 0; i < cloud->size(); ++i)
    {
        SuctionPoint point;
        point.position = cloud->points[i];
        point.normal.x = 0.0f;
        point.normal.y = 0.0f;
        point.normal.z = 1.0f;
        point.index = static_cast<int>(i);

        // 模拟评分
        point.scores.seal_score = 0.1f + (i % 10) * 0.1f;
        point.scores.wrench_score = 0.2f + (i % 8) * 0.1f;
        point.scores.feasibility_score = 0.3f + (i % 6) * 0.1f;
        point.scores.object_size_score = 0.4f + (i % 4) * 0.1f;
        point.scores.composite_score =
            (point.scores.seal_score + point.scores.wrench_score + point.scores.feasibility_score +
             point.scores.object_size_score) /
            4.0f;

        suction_points.push_back(point);
    }

    result.setSuctionPoints(suction_points);
    result.setInferenceTime(123.45);

    // 测试统计计算
    auto stats = result.getStatistics();
    std::cout << "统计信息:" << std::endl;
    std::cout << "  输入点数: " << result.getInputSize() << std::endl;
    std::cout << "  输出点数: " << result.getOutputSize() << std::endl;
    std::cout << "  推理时间: " << result.getInferenceTime() << " ms" << std::endl;
    std::cout << "  平均综合评分: " << stats.mean_composite_score << std::endl;
    std::cout << "  最高综合评分: " << stats.max_composite_score << std::endl;
    std::cout << "  最低综合评分: " << stats.min_composite_score << std::endl;

    // 测试文件保存
    std::string test_dir = "./test_results/";

    if (result.saveToTXT(test_dir + "test_result.txt") &&
        result.saveToJSON(test_dir + "test_result.json") &&
        result.saveToCSV(test_dir + "test_result.csv"))
    {
        std::cout << "文件保存测试: 通过" << std::endl;
    }
    else
    {
        std::cout << "文件保存测试: 失败" << std::endl;
        return false;
    }

    return true;
}

// 测试推理器类（不需要实际模型）
bool testInferenceClass()
{
    std::cout << "\n=== 测试推理器类 ===" << std::endl;

    InferenceConfig config;
    config.num_points = 1000;
    config.use_gpu = false;  // 测试时使用CPU
    config.device = "cpu";

    // 使用一个不存在的模型路径进行测试
    DSNetInference inference("./test_model.pt", config);

    // 测试配置设置和获取
    auto current_config = inference.getConfig();
    std::cout << "配置测试:" << std::endl;
    std::cout << "  采样点数: " << current_config.num_points << std::endl;
    std::cout << "  设备类型: " << current_config.device << std::endl;
    std::cout << "  密封权重: " << current_config.score_weights.seal_weight << std::endl;

    // 测试初始化状态
    std::cout << "初始化状态: " << (inference.isInitialized() ? "已初始化" : "未初始化")
              << std::endl;

    // 注意：这里不调用initialize()，因为没有真实的模型文件

    return true;
}

// 测试工具函数
bool testUtilityFunctions()
{
    std::cout << "\n=== 测试工具函数 ===" << std::endl;

    // 创建测试点云
    auto cloud = createTestPointCloud();

    // 测试边界框计算
    auto bbox = getPointCloudBoundingBox(cloud);
    std::cout << "边界框测试:" << std::endl;
    std::cout << "  最小点: (" << bbox.first.x << ", " << bbox.first.y << ", " << bbox.first.z
              << ")" << std::endl;
    std::cout << "  最大点: (" << bbox.second.x << ", " << bbox.second.y << ", " << bbox.second.z
              << ")" << std::endl;

    // 测试点云归一化
    auto normalized_cloud = std::make_shared<PointCloud>(*cloud);
    normalizePointCloud(normalized_cloud);
    auto normalized_bbox = getPointCloudBoundingBox(normalized_cloud);
    std::cout << "归一化后边界框:" << std::endl;
    std::cout << "  最小点: (" << normalized_bbox.first.x << ", " << normalized_bbox.first.y << ", "
              << normalized_bbox.first.z << ")" << std::endl;
    std::cout << "  最大点: (" << normalized_bbox.second.x << ", " << normalized_bbox.second.y
              << ", " << normalized_bbox.second.z << ")" << std::endl;

    // 测试文件保存和加载
    std::string test_file = "./test_results/test_cloud.pcd";
    if (savePointCloudToFile(cloud, test_file))
    {
        std::cout << "点云保存测试: 通过" << std::endl;

        auto loaded_cloud = loadPointCloudFromFile(test_file);
        if (loaded_cloud && loaded_cloud->size() == cloud->size())
        {
            std::cout << "点云加载测试: 通过" << std::endl;
        }
        else
        {
            std::cout << "点云加载测试: 失败" << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "点云保存测试: 失败" << std::endl;
        return false;
    }

    return true;
}

int main()
{
    std::cout << "DSNet C++ 推理库测试程序" << std::endl;
    std::cout << "=========================" << std::endl;

    // 创建测试结果目录
    system("mkdir -p ./test_results");

    bool all_passed = true;

    // 运行测试
    all_passed &= testInferenceResult();
    all_passed &= testInferenceClass();
    all_passed &= testUtilityFunctions();

    std::cout << "\n=== 测试总结 ===" << std::endl;
    if (all_passed)
    {
        std::cout << "✓ 所有测试通过!" << std::endl;

#ifdef USE_LIBTORCH
        std::cout << "\n编译配置:" << std::endl;
        std::cout << "✓ LibTorch: 已启用" << std::endl;
#else
        std::cout << "\n编译配置:" << std::endl;
        std::cout << "✗ LibTorch: 未启用 (需要真实模型推理请启用LibTorch)" << std::endl;
#endif

#ifdef USE_CUDA
        std::cout << "✓ CUDA: 已启用" << std::endl;
#else
        std::cout << "✗ CUDA: 未启用" << std::endl;
#endif

#ifdef USE_OPENCV
        std::cout << "✓ OpenCV: 已启用" << std::endl;
#else
        std::cout << "✗ OpenCV: 未启用" << std::endl;
#endif

        return 0;
    }
    else
    {
        std::cout << "✗ 部分测试失败!" << std::endl;
        return 1;
    }
}
