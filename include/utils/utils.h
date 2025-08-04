#ifndef DSNET_UTILS_H
#define DSNET_UTILS_H

#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace utils
{

// 直接使用PCL点云类型作为主要类型
typedef pcl::PointXYZ Point3D;
typedef pcl::PointNormal PointNormal;
typedef pcl::PointCloud<Point3D> PointCloud;
typedef pcl::PointCloud<PointNormal> PointCloudNormal;

}  // namespace utils

#endif  // DSNET_UTILS_H