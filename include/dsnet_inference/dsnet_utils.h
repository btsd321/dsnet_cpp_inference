#ifndef DSNET_UTILS_H
#define DSNET_UTILS_H

#ifdef USE_PCL
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#else
// Fallback minimal point cloud definitions
struct PointXYZ
{
    float x, y, z;
    PointXYZ() = default;
    PointXYZ(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};
template <typename T>
struct PointCloud
{
    std::vector<T> points;
};
#endif
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace dsnet
{

// 直接使用PCL点云类型作为主要类型
typedef pcl::PointXYZ Point3D;
typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointCloud<Point3D> PointCloud;
typedef pcl::PointCloud<PointNormal> PointCloudNormal;

PointCloud::Ptr loadPointCloudFromFile(const std::string& filename);

bool savePointCloudToFile(const PointCloud::Ptr& cloud, const std::string& filename);

void normalizePointCloud(PointCloud::Ptr& cloud);

std::pair<Point3D, Point3D> getPointCloudBoundingBox(const PointCloud::Ptr& cloud);

}  // namespace dsnet

#endif  // DSNET_UTILS_H