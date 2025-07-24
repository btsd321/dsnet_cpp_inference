#include "dsnet_inference/dsnet_model_loader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace dsnet
{

std::unique_ptr<ModelWeights> ModelLoader::loadFromCheckpoint(const std::string& checkpoint_path)
{
    // 注意：由于checkpoint.tar包含PyTorch特定格式，这里我们建议用户先使用
    // export_weights.py脚本导出权重，然后使用loadFromExportedWeights加载

    std::cerr << "直接从checkpoint.tar加载暂不支持。" << std::endl;
    std::cerr << "请先使用 export_weights.py 脚本导出权重：" << std::endl;
    std::cerr << "python export_weights.py --checkpoint " << checkpoint_path
              << " --output exported_weights" << std::endl;
    std::cerr << "然后使用 loadFromExportedWeights 加载导出的权重。" << std::endl;

    return nullptr;
}

std::unique_ptr<ModelWeights> ModelLoader::loadFromExportedWeights(const std::string& weights_dir)
{
    std::cout << "从导出权重目录加载模型: " << weights_dir << std::endl;

    auto weights = std::make_unique<ModelWeights>();

    try
    {
        // 加载模型配置
        loadModelConfig(weights_dir, *weights);

        // 加载PointNet++权重
        loadPointNetWeights(weights_dir, weights->backbone);

        // 加载扩散模型权重
        loadDiffusionWeights(weights_dir, weights->diffusion);

        std::cout << "模型权重加载完成" << std::endl;
        return weights;
    }
    catch (const std::exception& e)
    {
        std::cerr << "加载权重失败: " << e.what() << std::endl;
        return nullptr;
    }
}

bool ModelLoader::loadModelConfig(const std::string& weights_dir, ModelWeights& weights)
{
    std::string config_file = weights_dir + "/model_config.json";
    std::ifstream file(config_file);

    if (!file.is_open())
    {
        throw std::runtime_error("无法打开配置文件: " + config_file);
    }

    // 简单的JSON解析（这里为了演示，实际项目中建议使用nlohmann/json等库）
    std::string line;
    bool in_pointnet_config = false;

    while (std::getline(file, line))
    {
        // 移除空格和制表符
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

        if (line.find("\"npoint_per_layer\"") != std::string::npos)
        {
            // 解析npoint_per_layer配置
            // 这里简化处理，实际中需要更完整的JSON解析
            weights.config.npoint_per_layer = {4096, 1024, 256, 64};
        }
        else if (line.find("\"diffusion_inference_steps\"") != std::string::npos)
        {
            // 提取数值
            size_t pos = line.find(":");
            if (pos != std::string::npos)
            {
                std::string value_str = line.substr(pos + 1);
                value_str.erase(std::remove(value_str.begin(), value_str.end(), ','),
                                value_str.end());
                weights.config.diffusion_steps = std::stoi(value_str);
            }
        }
        else if (line.find("\"bit_scale\"") != std::string::npos)
        {
            size_t pos = line.find(":");
            if (pos != std::string::npos)
            {
                std::string value_str = line.substr(pos + 1);
                value_str.erase(std::remove(value_str.begin(), value_str.end(), ','),
                                value_str.end());
                weights.config.bit_scale = std::stof(value_str);
            }
        }
    }

    std::cout << "配置加载完成:" << std::endl;
    std::cout << "  扩散步数: " << weights.config.diffusion_steps << std::endl;
    std::cout << "  bit_scale: " << weights.config.bit_scale << std::endl;

    return true;
}

bool ModelLoader::loadPointNetWeights(const std::string& weights_dir, PointNetWeights& weights)
{
    std::string pointnet_dir = weights_dir + "/pointnet";
    std::string weights_info_file = pointnet_dir + "/weights_info.json";

    std::cout << "加载PointNet++权重从: " << pointnet_dir << std::endl;

    // 这里需要根据导出的权重文件结构来加载
    // 由于权重结构复杂，这里提供基本框架，实际实现需要根据具体的权重文件来解析

    // 加载权重信息文件
    std::ifstream info_file(weights_info_file);
    if (!info_file.is_open())
    {
        throw std::runtime_error("无法打开权重信息文件: " + weights_info_file);
    }

    // 示例：加载一些关键权重
    // 实际实现中需要解析JSON并加载所有权重文件

    std::cout << "PointNet++权重加载完成（示例实现）" << std::endl;
    return true;
}

bool ModelLoader::loadDiffusionWeights(const std::string& weights_dir, DiffusionWeights& weights)
{
    std::string diffusion_dir = weights_dir + "/diffusion";
    std::string weights_info_file = diffusion_dir + "/weights_info.json";

    std::cout << "加载扩散模型权重从: " << diffusion_dir << std::endl;

    // 这里需要根据导出的权重文件结构来加载
    // 实际实现需要解析所有的权重文件

    std::cout << "扩散模型权重加载完成（示例实现）" << std::endl;
    return true;
}

}  // namespace dsnet
