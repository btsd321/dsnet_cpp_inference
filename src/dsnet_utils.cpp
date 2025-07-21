#include "dsnet_inference/dsnet_utils.h"

namespace dsnet
{

// ==================== 工具函数实现 ====================

PointCloud::Ptr loadPointCloudFromFile(const std::string& filename)
{
    PointCloud::Ptr cloud(new PointCloud);

    // 获取文件扩展名
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == "pcd")
    {
        if (pcl::io::loadPCDFile(filename, *cloud) == -1)
        {
            std::cerr << "无法打开PCD文件: " << filename << std::endl;
            return nullptr;
        }
        std::cout << "加载PCD点云: " << cloud->points.size() << " 个点" << std::endl;
    }
    else if (extension == "ply")
    {
        if (pcl::io::loadPLYFile(filename, *cloud) == -1)
        {
            std::cerr << "无法打开PLY文件: " << filename << std::endl;
            return nullptr;
        }
        std::cout << "加载PLY点云: " << cloud->points.size() << " 个点" << std::endl;
    }
    else
    {
        // 传统的文本格式加载
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return nullptr;
        }

        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            float x, y, z;

            if (!(iss >> x >> y >> z))
            {
                continue;  // 跳过无效行
            }

            Point3D point;
            point.x = x;
            point.y = y;
            point.z = z;
            cloud->points.push_back(point);
        }

        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;

        std::cout << "加载点云: " << cloud->points.size() << " 个点" << std::endl;
    }

    return cloud;
}

bool savePointCloudToFile(const PointCloud::Ptr& cloud, const std::string& filename)
{
    // 获取文件扩展名
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == "pcd")
    {
        return pcl::io::savePCDFile(filename, *cloud) == 0;
    }
    else if (extension == "ply")
    {
        return pcl::io::savePLYFile(filename, *cloud) == 0;
    }
    else
    {
        // 保存为文本格式
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "无法创建文件: " << filename << std::endl;
            return false;
        }

        for (const auto& point : cloud->points)
        {
            file << point.x << " " << point.y << " " << point.z << "\n";
        }

        return true;
    }
}

void normalizePointCloud(PointCloud::Ptr& cloud)
{
    if (cloud->points.empty())
        return;

    // 计算质心
    Eigen::Vector3f centroid(0, 0, 0);
    for (const auto& point : cloud->points)
    {
        centroid += Eigen::Vector3f(point.x, point.y, point.z);
    }
    centroid /= static_cast<float>(cloud->points.size());

    // 移动到原点
    for (auto& point : cloud->points)
    {
        point.x -= centroid.x();
        point.y -= centroid.y();
        point.z -= centroid.z();
    }

    // 计算最大距离
    float max_distance = 0.0f;
    for (const auto& point : cloud->points)
    {
        float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        max_distance = std::max(max_distance, distance);
    }

    // 缩放到单位球
    if (max_distance > 1e-8f)
    {
        for (auto& point : cloud->points)
        {
            point.x /= max_distance;
            point.y /= max_distance;
            point.z /= max_distance;
        }
    }
}

std::pair<Point3D, Point3D> getPointCloudBoundingBox(const PointCloud::Ptr& cloud)
{
    Point3D min_pt, max_pt;

    if (cloud->points.empty())
    {
        min_pt.x = min_pt.y = min_pt.z = 0.0f;
        max_pt.x = max_pt.y = max_pt.z = 0.0f;
        return {min_pt, max_pt};
    }

    min_pt = max_pt = cloud->points[0];

    for (const auto& point : cloud->points)
    {
        min_pt.x = std::min(min_pt.x, point.x);
        min_pt.y = std::min(min_pt.y, point.y);
        min_pt.z = std::min(min_pt.z, point.z);

        max_pt.x = std::max(max_pt.x, point.x);
        max_pt.y = std::max(max_pt.y, point.y);
        max_pt.z = std::max(max_pt.z, point.z);
    }

    return {min_pt, max_pt};
}

}  // namespace dsnet