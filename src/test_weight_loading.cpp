#include <iostream>

#include "dsnet_inference/dsnet_model_loader.h"

int main()
{
    const std::string weightsDir =
        "/home/lixinlong/Project/dsnet_cpp_inference/pyinference/example/input/model/20250720";

    std::cout << "=== DSNet C++ 权重加载测试 ===" << std::endl;
    std::cout << "权重目录: " << weightsDir << std::endl;

    // 测试权重元数据解析
    std::map<std::string, dsnet::WeightInfo> weightMap;
    std::string weightsInfoPath = weightsDir + "/pointnet/weights_info.json";

    if (!dsnet::ModelLoader::parseWeightMetadata(weightsInfoPath, weightMap))
    {
        std::cerr << "权重元数据解析失败" << std::endl;
        return -1;
    }

    std::cout << "成功解析 " << weightMap.size() << " 个权重条目" << std::endl;

    // 测试加载几个示例权重
    std::vector<std::string> testKeys = {"SA_modules.0.mlps.0.layer0.conv.weight",
                                         "SA_modules.0.mlps.0.layer0.normlayer.bn.weight",
                                         "SA_modules.0.mlps.0.layer0.normlayer.bn.bias"};

    for (const auto& key : testKeys)
    {
        auto it = weightMap.find(key);
        if (it == weightMap.end())
        {
            std::cout << "权重 " << key << " 未找到" << std::endl;
            continue;
        }

        const auto& info = it->second;
        std::string binaryPath = weightsDir + "/pointnet/" + info.filename;

        Eigen::MatrixXf tensor;
        if (dsnet::ModelLoader::loadBinaryTensor(binaryPath, info.shape, tensor))
        {
            std::cout << "✓ 成功加载权重: " << key << std::endl;
            std::cout << "  形状: [";
            for (size_t i = 0; i < info.shape.size(); ++i)
            {
                std::cout << info.shape[i];
                if (i < info.shape.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  Eigen矩阵: " << tensor.rows() << "x" << tensor.cols() << std::endl;

            // 显示一些统计信息
            float minVal = tensor.minCoeff();
            float maxVal = tensor.maxCoeff();
            float meanVal = tensor.mean();
            std::cout << "  数值范围: [" << minVal << ", " << maxVal << "], 均值: " << meanVal
                      << std::endl;
        }
        else
        {
            std::cout << "✗ 加载权重失败: " << key << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "=== 权重加载测试完成 ===" << std::endl;
    return 0;
}
