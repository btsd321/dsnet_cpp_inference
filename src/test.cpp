#include <chrono>
#include <iostream>
#include <random>

#include "dsnet_inference/dsnet_inference.h"

using namespace dsnet;

/**
 * @brief 创建测试点云数据
 */
PointCloud createTestPointCloud(int num_points = 2500)
{
    PointCloud cloud;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);

    // 创建立方体点云
    for (int i = 0; i < num_points / 3; ++i)
    {
        Point3D point(pos_dist(gen) * 0.5f, pos_dist(gen) * 0.5f, pos_dist(gen) * 0.5f);

        Point3D normal(normal_dist(gen), normal_dist(gen), normal_dist(gen));

        // 归一化法向量
        float norm = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (norm > 1e-6f)
        {
            normal.x /= norm;
            normal.y /= norm;
            normal.z /= norm;
        }

        cloud.addPoint(point, normal);
    }

    // 创建球体点云
    for (int i = 0; i < num_points / 3; ++i)
    {
        float theta = pos_dist(gen) * M_PI;
        float phi = pos_dist(gen) * 2 * M_PI;
        float r = 0.3f;

        Point3D point(r * std::sin(theta) * std::cos(phi) + 1.0f,
                      r * std::sin(theta) * std::sin(phi), r * std::cos(theta));

        // 球体的法向量指向外部
        Point3D normal(point.x - 1.0f, point.y, point.z);

        float norm = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (norm > 1e-6f)
        {
            normal.x /= norm;
            normal.y /= norm;
            normal.z /= norm;
        }

        cloud.addPoint(point, normal);
    }

    // 创建平面点云
    for (int i = 0; i < num_points / 3; ++i)
    {
        Point3D point(pos_dist(gen), pos_dist(gen), -0.8f);

        Point3D normal(0.0f, 0.0f, 1.0f);  // 向上的法向量

        cloud.addPoint(point, normal);
    }

    return cloud;
}

/**
 * @brief 测试基本功能
 */
bool testBasicFunctionality()
{
    std::cout << "\n=== 测试基本功能 ===" << std::endl;

    // 创建测试点云
    auto test_cloud = createTestPointCloud(1000);
    std::cout << "创建测试点云: " << test_cloud.size() << " 个点" << std::endl;

    // 测试保存和加载
    std::string temp_file = "/tmp/test_cloud.txt";
    if (test_cloud.saveToFile(temp_file))
    {
        std::cout << "✓ 点云保存成功" << std::endl;
    }
    else
    {
        std::cout << "✗ 点云保存失败" << std::endl;
        return false;
    }

    PointCloud loaded_cloud;
    if (loaded_cloud.loadFromFile(temp_file))
    {
        std::cout << "✓ 点云加载成功: " << loaded_cloud.size() << " 个点" << std::endl;
    }
    else
    {
        std::cout << "✗ 点云加载失败" << std::endl;
        return false;
    }

    // 测试归一化
    auto bbox_before = test_cloud.getBoundingBox();
    test_cloud.normalize();
    auto bbox_after = test_cloud.getBoundingBox();

    std::cout << "归一化前边界框: (" << bbox_before.first.x << "," << bbox_before.first.y << ","
              << bbox_before.first.z << ") - (" << bbox_before.second.x << ","
              << bbox_before.second.y << "," << bbox_before.second.z << ")" << std::endl;
    std::cout << "归一化后边界框: (" << bbox_after.first.x << "," << bbox_after.first.y << ","
              << bbox_after.first.z << ") - (" << bbox_after.second.x << "," << bbox_after.second.y
              << "," << bbox_after.second.z << ")" << std::endl;

    return true;
}

/**
 * @brief 测试推理功能
 */
bool testInference()
{
    std::cout << "\n=== 测试推理功能 ===" << std::endl;

    // 配置推理参数
    InferenceConfig config;
    config.num_points = 1024;
    config.diffusion_steps = 20;
    config.use_gpu = false;  // 测试时使用CPU

    // 创建推理器（使用占位符模型路径）
    DSNetInference inferencer("dummy_model.pth", config);

    // 初始化
    if (!inferencer.initialize())
    {
        std::cout << "✗ 推理器初始化失败" << std::endl;
        return false;
    }
    std::cout << "✓ 推理器初始化成功" << std::endl;

    // 创建测试数据
    auto test_cloud = createTestPointCloud(2500);

    // 执行推理
    try
    {
        auto result = inferencer.predict(test_cloud);

        std::cout << "✓ 推理成功完成" << std::endl;
        std::cout << "  - 推理时间: " << result.inference_time_ms << " ms" << std::endl;
        std::cout << "  - 预处理后点数: " << result.preprocessed_cloud.size() << std::endl;
        std::cout << "  - 评分数量: " << result.scores.size() << std::endl;
        std::cout << "  - 最佳点数量: " << result.best_points.size() << std::endl;

        // 显示前5个最佳点
        std::cout << "\n前5个最佳吸取点:" << std::endl;
        for (size_t i = 0; i < std::min(5ul, result.best_points.size()); ++i)
        {
            const auto& point = result.best_points[i];
            std::cout << "  第" << (i + 1) << "名: 位置(" << point.position.x << ", "
                      << point.position.y << ", " << point.position.z
                      << "), 综合评分: " << point.scores.composite_score << std::endl;
        }

        // 保存结果
        if (saveInferenceResult(result, "/tmp/test_inference_result.txt"))
        {
            std::cout << "✓ 结果保存成功" << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "✗ 推理失败: " << e.what() << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief 测试批量推理
 */
bool testBatchInference()
{
    std::cout << "\n=== 测试批量推理 ===" << std::endl;

    InferenceConfig config;
    config.num_points = 512;
    config.diffusion_steps = 10;
    config.use_gpu = false;

    DSNetInference inferencer("dummy_model.pth", config);

    if (!inferencer.initialize())
    {
        std::cout << "✗ 推理器初始化失败" << std::endl;
        return false;
    }

    // 创建多个测试点云
    std::vector<PointCloud> test_clouds;
    for (int i = 0; i < 3; ++i)
    {
        test_clouds.push_back(createTestPointCloud(500 + i * 200));
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行批量推理
    try
    {
        auto results = inferencer.predictBatch(test_clouds);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "✓ 批量推理成功完成" << std::endl;
        std::cout << "  - 总用时: " << duration.count() << " ms" << std::endl;
        std::cout << "  - 场景数量: " << results.size() << std::endl;

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << "  - 场景" << (i + 1) << ": " << results[i].inference_time_ms << " ms, "
                      << results[i].best_points.size() << " 个最佳点" << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "✗ 批量推理失败: " << e.what() << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief 性能测试
 */
bool testPerformance()
{
    std::cout << "\n=== 性能测试 ===" << std::endl;

    InferenceConfig config;
    config.num_points = 16384;
    config.diffusion_steps = 50;
    config.use_gpu = false;

    DSNetInference inferencer("dummy_model.pth", config);

    if (!inferencer.initialize())
    {
        std::cout << "✗ 推理器初始化失败" << std::endl;
        return false;
    }

    // 测试不同大小的点云
    std::vector<int> test_sizes = {1000, 5000, 10000, 20000};

    for (int size : test_sizes)
    {
        auto test_cloud = createTestPointCloud(size);

        auto start_time = std::chrono::high_resolution_clock::now();

        try
        {
            auto result = inferencer.predict(test_cloud);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            std::cout << "  点云大小: " << size << " -> 推理时间: " << duration.count() << " ms"
                      << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cout << "  点云大小: " << size << " -> 推理失败: " << e.what() << std::endl;
        }
    }

    return true;
}

int main()
{
    std::cout << "DSNet C++ 推理系统测试程序" << std::endl;
    std::cout << "==============================" << std::endl;

    bool all_passed = true;

    // 运行各项测试
    all_passed &= testBasicFunctionality();
    all_passed &= testInference();
    all_passed &= testBatchInference();
    all_passed &= testPerformance();

    std::cout << "\n==============================" << std::endl;
    if (all_passed)
    {
        std::cout << "🎉 所有测试通过！" << std::endl;
        std::cout << "DSNet C++ 推理系统工作正常" << std::endl;
    }
    else
    {
        std::cout << "❌ 部分测试失败" << std::endl;
        std::cout << "请检查错误信息并修复问题" << std::endl;
    }

    std::cout << "\n使用说明:" << std::endl;
    std::cout << "1. 安装依赖: Eigen3, OpenCV (可选), LibTorch (可选)" << std::endl;
    std::cout << "2. 编译项目: mkdir build && cd build && cmake .. && make" << std::endl;
    std::cout << "3. 运行测试: ./bin/dsnet_test" << std::endl;
    std::cout << "4. 集成到你的项目中使用 DSNetInference 类" << std::endl;

    return all_passed ? 0 : 1;
}