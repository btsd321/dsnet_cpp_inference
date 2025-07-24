#include <chrono>
#include <iostream>
#include <random>

#include "dsnet_inference/dsnet_pointnet_backbone.h"

using namespace dsnet;

int main()
{
    std::cout << "=== PointNet++ 算法测试 ===" << std::endl;

    // 创建测试点云数据 (3 x N)
    const int num_points = 1000;
    Eigen::MatrixXf points(3, num_points);

    // 生成随机点云
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < num_points; ++i)
    {
        points(0, i) = dis(gen);  // x
        points(1, i) = dis(gen);  // y
        points(2, i) = dis(gen);  // z
    }

    std::cout << "生成测试点云: " << points.cols() << " 个点" << std::endl;

    // 测试1: FarthestPointSampling
    std::cout << "\n--- 测试 FarthestPointSampling ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    auto sampled_indices = FarthestPointSampling::sample(points, 256);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "FPS采样结果: " << sampled_indices.size() << " 个点" << std::endl;
    std::cout << "FPS耗时: " << duration.count() << " ms" << std::endl;

    if (sampled_indices.size() == 256)
    {
        std::cout << "✓ FPS测试通过" << std::endl;
    }
    else
    {
        std::cout << "✗ FPS测试失败" << std::endl;
        return 1;
    }

    // 测试2: BallQuery
    std::cout << "\n--- 测试 BallQuery ---" << std::endl;

    // 创建查询点 (取FPS采样的前10个点)
    Eigen::MatrixXf query_points(3, 10);
    for (int i = 0; i < 10; ++i)
    {
        query_points.col(i) = points.col(sampled_indices[i]);
    }

    start = std::chrono::high_resolution_clock::now();

    auto grouped_indices = BallQuery::query(query_points, points, 0.3f, 32);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Ball Query结果: " << grouped_indices.size() << " 个查询点" << std::endl;
    std::cout << "Ball Query耗时: " << duration.count() << " ms" << std::endl;

    bool ball_query_success = true;
    for (size_t i = 0; i < grouped_indices.size(); ++i)
    {
        if (grouped_indices[i].size() != 32)
        {
            ball_query_success = false;
            break;
        }
    }

    if (ball_query_success)
    {
        std::cout << "✓ Ball Query测试通过" << std::endl;
    }
    else
    {
        std::cout << "✗ Ball Query测试失败" << std::endl;
        return 1;
    }

    // 测试3: MLPLayer (创建简单的测试权重)
    std::cout << "\n--- 测试 MLPLayer ---" << std::endl;

    // 创建测试MLP权重
    MLPWeights mlp_weights;

    // 第一层: 3->64
    ConvWeights conv1;
    conv1.weight = Eigen::MatrixXf::Random(64, 3);
    conv1.bias = Eigen::VectorXf::Random(64);
    conv1.has_bias = true;

    NormWeights norm1;
    norm1.weight = Eigen::VectorXf::Ones(64);
    norm1.bias = Eigen::VectorXf::Zero(64);
    norm1.running_mean = Eigen::VectorXf::Zero(64);
    norm1.running_var = Eigen::VectorXf::Ones(64);
    norm1.eps = 1e-5f;

    mlp_weights.conv_layers.push_back(conv1);
    mlp_weights.norm_layers.push_back(norm1);
    mlp_weights.has_norm.push_back(true);

    MLPLayer mlp(mlp_weights);

    // 测试输入 (3 x 100)
    Eigen::MatrixXf test_input = Eigen::MatrixXf::Random(3, 100);

    start = std::chrono::high_resolution_clock::now();

    auto mlp_output = mlp.forward(test_input);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "MLP输入形状: " << test_input.rows() << " x " << test_input.cols() << std::endl;
    std::cout << "MLP输出形状: " << mlp_output.rows() << " x " << mlp_output.cols() << std::endl;
    std::cout << "MLP耗时: " << duration.count() << " ms" << std::endl;

    if (mlp_output.rows() == 64 && mlp_output.cols() == 100)
    {
        std::cout << "✓ MLP测试通过" << std::endl;
    }
    else
    {
        std::cout << "✗ MLP测试失败" << std::endl;
        return 1;
    }

    std::cout << "\n=== 所有PointNet++算法测试通过! ===" << std::endl;

    return 0;
}
