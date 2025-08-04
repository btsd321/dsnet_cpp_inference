// #include <fstream>
// #include <iostream>
// #include <sstream>

// #include "dsnet_inference/dsnet_model_loader.h"

// namespace dsnet
// {

// bool ModelLoader::loadBinaryTensor(const std::string& filePath, const std::vector<int>& shape,
//                                    Eigen::MatrixXf& tensor)
// {
//     std::ifstream file(filePath, std::ios::binary);
//     if (!file.is_open())
//     {
//         std::cerr << "Cannot open binary file: " << filePath << std::endl;
//         return false;
//     }

//     // Calculate total elements
//     int totalElements = 1;
//     for (int dim : shape)
//     {
//         totalElements *= dim;
//     }

//     // Read binary data as float32
//     std::vector<float> data(totalElements);
//     file.read(reinterpret_cast<char*>(data.data()), totalElements * sizeof(float));

//     if (!file.good())
//     {
//         std::cerr << "Failed to read data from: " << filePath << std::endl;
//         file.close();
//         return false;
//     }

//     // Reshape to matrix based on shape
//     if (shape.size() == 4)
//     {
//         // Conv weight: [out_channels, in_channels, kernel_h, kernel_w]
//         // Flatten to [out_channels, in_channels * kernel_h * kernel_w]
//         int out_channels = shape[0];
//         int feature_size = shape[1] * shape[2] * shape[3];
//         tensor = Eigen::Map<Eigen::MatrixXf>(data.data(), feature_size, out_channels);
//     }
//     else if (shape.size() == 1)
//     {
//         // Bias/BN parameters: [channels]
//         tensor = Eigen::Map<Eigen::MatrixXf>(data.data(), shape[0], 1);
//     }
//     else if (shape.size() == 2)
//     {
//         // Linear layer: [out_features, in_features]
//         tensor = Eigen::Map<Eigen::MatrixXf>(data.data(), shape[1], shape[0]);
//     }
//     else
//     {
//         // General case: flatten to column vector
//         tensor = Eigen::Map<Eigen::MatrixXf>(data.data(), totalElements, 1);
//     }

//     file.close();
//     std::cout << "Loaded tensor from " << filePath << " with shape [";
//     for (size_t i = 0; i < shape.size(); ++i)
//     {
//         std::cout << shape[i];
//         if (i < shape.size() - 1)
//             std::cout << ", ";
//     }
//     std::cout << "] -> Eigen matrix " << tensor.rows() << "x" << tensor.cols() << std::endl;

//     return true;
// }

// bool ModelLoader::loadTextTensor(const std::string& filePath, const std::vector<int>& shape,
//                                  Eigen::MatrixXf& tensor)
// {
//     std::ifstream file(filePath);
//     if (!file.is_open())
//     {
//         std::cerr << "Cannot open text file: " << filePath << std::endl;
//         return false;
//     }

//     // Calculate total elements
//     int totalElements = 1;
//     for (int dim : shape)
//     {
//         totalElements *= dim;
//     }

//     std::vector<float> data;
//     data.reserve(totalElements);

//     std::string line;
//     while (std::getline(file, line) && data.size() < totalElements)
//     {
//         if (!line.empty())
//         {
//             try
//             {
//                 float value = std::stof(line);
//                 data.push_back(value);
//             }
//             catch (const std::exception& e)
//             {
//                 std::cerr << "Failed to parse float from line: " << line << std::endl;
//                 return false;
//             }
//         }
//     }

//     if (data.size() != totalElements)
//     {
//         std::cerr << "Expected " << totalElements << " elements, got " << data.size() <<
//         std::endl; return false;
//     }

//     // Reshape similar to binary tensor
//     if (shape.size() == 4)
//     {
//         int out_channels = shape[0];
//         int feature_size = shape[1] * shape[2] * shape[3];
//         tensor = Eigen::Map<Eigen::MatrixXf>(data.data(), feature_size, out_channels);
//     }
//     else if (shape.size() == 1)
//     {
//         tensor = Eigen::Map<Eigen::MatrixXf>(data.data(), shape[0], 1);
//     }
//     else if (shape.size() == 2)
//     {
//         tensor = Eigen::Map<Eigen::MatrixXf>(data.data(), shape[1], shape[0]);
//     }
//     else
//     {
//         tensor = Eigen::Map<Eigen::MatrixXf>(data.data(), totalElements, 1);
//     }

//     file.close();
//     return true;
// }

// // Helper function to parse JSON weight metadata
// bool ModelLoader::parseWeightMetadata(const std::string& weightsInfoPath,
//                                       std::map<std::string, WeightInfo>& weightMap)
// {
//     std::ifstream file(weightsInfoPath);
//     if (!file.is_open())
//     {
//         std::cerr << "Cannot open weights info file: " << weightsInfoPath << std::endl;
//         return false;
//     }

//     std::string content((std::istreambuf_iterator<char>(file)),
//     std::istreambuf_iterator<char>()); file.close();

//     // Simple JSON parsing for weight metadata
//     // In production, use a proper JSON library like nlohmann/json
//     size_t pos = 0;
//     while ((pos = content.find("\"", pos)) != std::string::npos)
//     {
//         size_t keyStart = pos + 1;
//         size_t keyEnd = content.find("\"", keyStart);
//         if (keyEnd == std::string::npos)
//             break;

//         std::string key = content.substr(keyStart, keyEnd - keyStart);

//         // Skip non-weight keys
//         if (key.find("SA_modules") == std::string::npos &&
//             key.find("FP_modules") == std::string::npos)
//         {
//             pos = keyEnd + 1;
//             continue;
//         }

//         WeightInfo info;

//         // Find shape array
//         size_t shapeStart = content.find("\"shape\"", keyEnd);
//         if (shapeStart == std::string::npos)
//         {
//             pos = keyEnd + 1;
//             continue;
//         }

//         size_t arrayStart = content.find("[", shapeStart);
//         size_t arrayEnd = content.find("]", arrayStart);
//         if (arrayStart == std::string::npos || arrayEnd == std::string::npos)
//         {
//             pos = keyEnd + 1;
//             continue;
//         }

//         std::string shapeStr = content.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
//         std::stringstream ss(shapeStr);
//         std::string item;
//         while (std::getline(ss, item, ','))
//         {
//             // Remove whitespace
//             item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
//             if (!item.empty())
//             {
//                 info.shape.push_back(std::stoi(item));
//             }
//         }

//         // Find file name
//         size_t fileStart = content.find("\"file\"", arrayEnd);
//         if (fileStart != std::string::npos)
//         {
//             size_t fileNameStart = content.find("\"", fileStart + 6);
//             size_t fileNameEnd = content.find("\"", fileNameStart + 1);
//             if (fileNameStart != std::string::npos && fileNameEnd != std::string::npos)
//             {
//                 info.filename = content.substr(fileNameStart + 1, fileNameEnd - fileNameStart -
//                 1);
//             }
//         }

//         weightMap[key] = info;
//         pos = keyEnd + 1;
//     }

//     std::cout << "Parsed " << weightMap.size() << " weight entries from metadata" << std::endl;
//     return !weightMap.empty();
// }

// }  // namespace dsnet
