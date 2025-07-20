#!/bin/bash

# DSNet C++ 推理系统构建脚本

set -e

echo "=========================================="
echo "DSNet C++ 推理系统构建脚本"
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

echo "项目目录: ${PROJECT_DIR}"

# 检查是否安装了必要的依赖
echo "检查依赖..."

# 检查CMake
CMAKE_CMD=""
if command -v cmake >/dev/null 2>&1; then
    CMAKE_CMD="cmake"
elif [ -f "/usr/bin/cmake" ]; then
    CMAKE_CMD="/usr/bin/cmake"
elif [ -f "/usr/local/bin/cmake" ]; then
    CMAKE_CMD="/usr/local/bin/cmake"
else
    echo "错误: CMake 未安装"
    echo "请安装CMake: sudo apt-get install cmake"
    exit 1
fi

# 检查编译器
if ! command -v g++ >/dev/null 2>&1; then
    echo "错误: g++ 编译器未安装"
    echo "请安装g++: sudo apt-get install build-essential"
    exit 1
fi

echo "✓ CMake: $(${CMAKE_CMD} --version | head -n1)"
echo "✓ 编译器: $(g++ --version | head -n1)"

# 检查Eigen3
if ! pkg-config --exists eigen3 2>/dev/null; then
    echo "警告: Eigen3 未找到，尝试安装..."
    echo "sudo apt-get install libeigen3-dev"
fi

# 检查OpenCV (可选)
if pkg-config --exists opencv4 2>/dev/null; then
    echo "✓ OpenCV: $(pkg-config --modversion opencv4)"
else
    echo "注意: OpenCV 未找到 (可选依赖)"
    echo "如需可视化功能，请安装: sudo apt-get install libopencv-dev"
fi

# 创建构建目录
BUILD_DIR="${PROJECT_DIR}/build"
echo "创建构建目录: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 配置CMake
echo "配置CMake..."
CMAKE_OPTIONS=""

# 检查构建类型参数
BUILD_TYPE="Release"
if [ "$1" == "debug" ]; then
    BUILD_TYPE="Debug"
    echo "使用Debug构建"
elif [ "$1" == "release" ]; then
    BUILD_TYPE="Release"
    echo "使用Release构建"
fi

CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"

# 检查是否有vcpkg
if [ -d "${HOME}/SoftWare/vcpkg" ]; then
    echo "检测到vcpkg，使用vcpkg工具链"
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_TOOLCHAIN_FILE=${HOME}/SoftWare/vcpkg/scripts/buildsystems/vcpkg.cmake"
fi

# 运行CMake配置
cmake ${CMAKE_OPTIONS} "${PROJECT_DIR}"

# 编译项目
echo "开始编译..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "构建完成！"
echo "=========================================="

# 显示生成的文件
echo "生成的文件:"
ls -la "${BUILD_DIR}/bin/" 2>/dev/null || echo "  没有可执行文件生成"
ls -la "${BUILD_DIR}/lib/" 2>/dev/null || echo "  没有库文件生成"

echo ""
echo "运行测试:"
echo "  cd ${BUILD_DIR}"
echo "  ./bin/dsnet_test"
echo ""

# 自动运行测试
if [ "$2" == "test" ] || [ "$1" == "test" ]; then
    echo "自动运行测试..."
    if [ -f "${BUILD_DIR}/bin/dsnet_test" ]; then
        "${BUILD_DIR}/bin/dsnet_test"
    else
        echo "测试可执行文件不存在"
    fi
fi

echo "构建脚本完成！"
