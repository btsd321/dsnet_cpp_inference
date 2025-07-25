cmake_minimum_required(VERSION 3.16)
project(dsnet_inference VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 如果有vcpkg，使用vcpkg工具链
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    message(STATUS "Using vcpkg toolchain from VCPKG_ROOT")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    message(STATUS "Using local vcpkg toolchain")
endif()

# 设置编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -DDEBUG")
    message(STATUS "Debug模式: 启用调试信息，禁用优化")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -DNDEBUG")
    message(STATUS "Release模式: 启用最高优化，禁用调试信息")
else()
    # 默认Release模式
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -DNDEBUG")
    message(STATUS "默认Release模式: 启用最高优化")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 查找依赖包
find_package(Eigen3 REQUIRED)

# 设置LibTorch路径
set(LIBTORCH_ROOT "/home/lixinlong/Library/libtorch/install")
if(EXISTS "${LIBTORCH_ROOT}/share/cmake/Torch")
    set(CMAKE_PREFIX_PATH "${LIBTORCH_ROOT};${CMAKE_PREFIX_PATH}")
    message(STATUS "Using LibTorch from: ${LIBTORCH_ROOT}")
else()
    message(FATAL_ERROR "LibTorch not found at ${LIBTORCH_ROOT}. Please set LIBTORCH_ROOT to the correct path.")
endif()

# 可选依赖
find_package(OpenCV QUIET)
find_package(PCL CONFIG QUIET)
find_package(Torch QUIET)
find_package(CUDA QUIET)

# 设置编译标志
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    add_definitions(-DUSE_OPENCV)
else()
    message(WARNING "OpenCV not found - 某些功能可能不可用")
endif()

if(PCL_FOUND)
    message(STATUS "Found PCL: ${PCL_VERSION}")
    add_definitions(-DUSE_PCL)
else()
    message(WARNING "PCL not found - 点云功能可能不可用")
endif()

if(Torch_FOUND)
    message(STATUS "Found LibTorch: ${Torch_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    add_definitions(-DUSE_TORCH)
else()
    message(WARNING "LibTorch not found - 将使用手写实现")
endif()

if(CUDA_FOUND)
    message(STATUS "Found CUDA: ${CUDA_VERSION}")
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
else()
    message(WARNING "CUDA not found - 将使用CPU实现")
endif()

message(STATUS "LibTorch found: ${Torch_VERSION}")
message(STATUS "PCL found: ${PCL_VERSION}")
message(STATUS "CUDA found: ${CUDA_VERSION}")

# argparse是header-only库，直接包含vcpkg安装的路径
if(EXISTS "~/SoftWare/vcpkg/installed/x64-linux/include/argparse")
    set(ARGPARSE_INCLUDE_DIR "~/SoftWare/vcpkg/installed/x64-linux/include")
    message(STATUS "Found argparse in vcpkg")
endif()

# 收集pointnet2相关源文件
file(GLOB POINTNET2_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/pointnet2/*.cpp")
# 收集dsnet相关源文件
file(GLOB DSNET_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/dsnet/*.cpp")
# 收集pointnet2相关头文件
file(GLOB POINTNET2_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/pointnet2/*.h")
# 收集dsnet相关头文件
file(GLOB DSNET_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/dsnet_inference/*.h")
# 收集测试源文件
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/test/*.cpp")

# 创建dsnet推理库
set(PROJECT_SOURCES
    "${POINTNET2_SOURCES}"
    "${DSNET_SOURCES}"
)

set(PROJECT_HEADERS
    "${POINTNET2_HEADERS}"
    "${DSNET_HEADERS}"
)

# 创建静态库
add_library(dsnet_inference STATIC ${PROJECT_SOURCES})

# 基础链接库
target_link_libraries(dsnet_inference Eigen3::Eigen)

# 可选链接库
if(OpenCV_FOUND)
    target_link_libraries(dsnet_inference ${OpenCV_LIBS})
endif()

if(PCL_FOUND)
    target_link_libraries(dsnet_inference ${PCL_LIBRARIES})
endif()

if(Torch_FOUND)
    target_link_libraries(dsnet_inference ${TORCH_LIBRARIES})
endif()

if(CUDA_FOUND)
    target_link_libraries(dsnet_inference ${CUDA_LIBRARIES})
endif()

# 设置库的包含目录
target_include_directories(dsnet_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# 可选包含目录
if(OpenCV_FOUND)
    target_include_directories(dsnet_inference PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()

if(PCL_FOUND)
    target_include_directories(dsnet_inference PUBLIC ${PCL_INCLUDE_DIRS})
    # 手动添加vcpkg PCL路径作为备用
    target_include_directories(dsnet_inference PUBLIC "/home/lixinlong/SoftWare/vcpkg/installed/x64-linux/include")
    message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")
endif()

# 创建测试可执行文件
add_executable(dsnet_test 
    "${PROJECT_SOURCES}"
    "${TEST_SOURCES}"
)

# 安装规则
install(TARGETS dsnet_inference
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${PROJECT_HEADERS}
    DESTINATION include
)

# install(TARGETS dsnet_test
#     DESTINATION bin
# )

# 添加测试
enable_testing()

# 添加权重加载测试用例  
# add_test(NAME test_weight_loading COMMAND test_weight_loading)

# 打印配置信息
message(STATUS "")
message(STATUS "=================== DSNet C++ Inference Configuration ===================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 Version: ${Eigen3_VERSION}")
message(STATUS "LibTorch Version: ${Torch_VERSION}")
message(STATUS "PCL Version: ${PCL_VERSION}")
message(STATUS "Argparse Include: ${ARGPARSE_INCLUDE_DIR}")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================================================")
message(STATUS "")