// #include "dsnet_inference/dsnet_utils.h"

// namespace dsnet
// {

// // ==================== 工具函数实现 ====================

// bool savePointCloudToFile(const PointCloud::Ptr& cloud, const std::string& filename)
// {
//     // 获取文件扩展名
//     std::string extension = filename.substr(filename.find_last_of(".") + 1);
//     std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

//     if (extension == "pcd")
//     {
//         return pcl::io::savePCDFile(filename, *cloud) == 0;
//     }
//     else if (extension == "ply")
//     {
//         return pcl::io::savePLYFile(filename, *cloud) == 0;
//     }
//     else
//     {
//         // 保存为文本格式
//         std::ofstream file(filename);
//         if (!file.is_open())
//         {
//             std::cerr << "无法创建文件: " << filename << std::endl;
//             return false;
//         }

//         for (const auto& point : cloud->points)
//         {
//             file << point.x << " " << point.y << " " << point.z << "\n";
//         }

//         return true;
//     }
// }

// void normalizePointCloud(PointCloud::Ptr& cloud)
// {
//     if (cloud->points.empty())
//         return;

//     // 计算质心
//     Eigen::Vector3f centroid(0, 0, 0);
//     for (const auto& point : cloud->points)
//     {
//         centroid += Eigen::Vector3f(point.x, point.y, point.z);
//     }
//     centroid /= static_cast<float>(cloud->points.size());

//     // 移动到原点
//     for (auto& point : cloud->points)
//     {
//         point.x -= centroid.x();
//         point.y -= centroid.y();
//         point.z -= centroid.z();
//     }

//     // 计算最大距离
//     float max_distance = 0.0f;
//     for (const auto& point : cloud->points)
//     {
//         float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
//         max_distance = std::max(max_distance, distance);
//     }

//     // 缩放到单位球
//     if (max_distance > 1e-8f)
//     {
//         for (auto& point : cloud->points)
//         {
//             point.x /= max_distance;
//             point.y /= max_distance;
//             point.z /= max_distance;
//         }
//     }
// }

// std::pair<Point3D, Point3D> getPointCloudBoundingBox(const PointCloud::Ptr& cloud)
// {
//     Point3D min_pt, max_pt;

//     if (cloud->points.empty())
//     {
//         min_pt.x = min_pt.y = min_pt.z = 0.0f;
//         max_pt.x = max_pt.y = max_pt.z = 0.0f;
//         return {min_pt, max_pt};
//     }

//     min_pt = max_pt = cloud->points[0];

//     for (const auto& point : cloud->points)
//     {
//         min_pt.x = std::min(min_pt.x, point.x);
//         min_pt.y = std::min(min_pt.y, point.y);
//         min_pt.z = std::min(min_pt.z, point.z);

//         max_pt.x = std::max(max_pt.x, point.x);
//         max_pt.y = std::max(max_pt.y, point.y);
//         max_pt.z = std::max(max_pt.z, point.z);
//     }

//     return {min_pt, max_pt};
// }

// }  // namespace dsnet