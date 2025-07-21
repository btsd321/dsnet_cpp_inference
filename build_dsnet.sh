#!/bin/bash

# DSNet C++ 推理库构建脚本
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[BUILD]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 默认配置
BUILD_TYPE="Release"
USE_LIBTORCH="ON"
USE_CUDA="OFF"
USE_OPENCV="OFF"
LIBTORCH_DIR=""
BUILD_DIR="build"
CLEAN_BUILD="OFF"
VERBOSE="OFF"

# 帮助信息
show_help() {
    echo "DSNet C++ 推理库构建脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -t, --type TYPE         构建类型 (Debug|Release|RelWithDebInfo) [默认: Release]"
    echo "  -l, --libtorch DIR      LibTorch安装目录"
    echo "  -c, --cuda              启用CUDA支持"
    echo "  -o, --opencv            启用OpenCV支持"
    echo "  -b, --build-dir DIR     构建目录 [默认: build]"
    echo "  --clean                 清理构建目录"
    echo "  -v, --verbose           详细构建输出"
    echo "  --no-libtorch           禁用LibTorch支持"
    echo ""
    echo "示例:"
    echo "  $0 --libtorch /opt/libtorch"
    echo "  $0 --cuda --opencv --libtorch /opt/libtorch"
    echo "  $0 --type Debug --verbose"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -l|--libtorch)
            LIBTORCH_DIR="$2"
            shift 2
            ;;
        -c|--cuda)
            USE_CUDA="ON"
            shift
            ;;
        -o|--opencv)
            USE_OPENCV="ON"
            shift
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        -v|--verbose)
            VERBOSE="ON"
            shift
            ;;
        --no-libtorch)
            USE_LIBTORCH="OFF"
            shift
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查构建类型
if [[ ! "$BUILD_TYPE" =~ ^(Debug|Release|RelWithDebInfo)$ ]]; then
    print_error "无效的构建类型: $BUILD_TYPE"
    print_info "支持的构建类型: Debug, Release, RelWithDebInfo"
    exit 1
fi

print_message "DSNet C++ 推理库构建开始"
print_info "构建配置:"
print_info "  构建类型: $BUILD_TYPE"
print_info "  LibTorch: $USE_LIBTORCH"
print_info "  CUDA: $USE_CUDA"
print_info "  OpenCV: $USE_OPENCV"
print_info "  构建目录: $BUILD_DIR"

# 检查依赖
check_dependencies() {
    print_message "检查依赖..."
    
    # 检查CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake未安装"
        exit 1
    fi
    
    # 检查PCL
    if ! pkg-config --exists pcl_common-1.12 || ! pkg-config --exists pcl_io-1.12; then
        if ! pkg-config --exists pcl_common-1.10 || ! pkg-config --exists pcl_io-1.10; then
            print_warning "PCL库可能未安装，请确保已安装PCL开发包"
        fi
    fi
    
    # 检查Eigen3
    if ! pkg-config --exists eigen3; then
        print_warning "Eigen3库可能未安装，请确保已安装libeigen3-dev"
    fi
    
    # 检查LibTorch
    if [[ "$USE_LIBTORCH" == "ON" ]]; then
        if [[ -z "$LIBTORCH_DIR" ]]; then
            print_warning "未指定LibTorch目录，尝试自动检测..."
            # 尝试常见路径
            for path in "/opt/libtorch" "/usr/local/libtorch" "$HOME/libtorch" "./libtorch"; do
                if [[ -d "$path" && -f "$path/share/cmake/Torch/TorchConfig.cmake" ]]; then
                    LIBTORCH_DIR="$path"
                    print_info "找到LibTorch: $LIBTORCH_DIR"
                    break
                fi
            done
            
            if [[ -z "$LIBTORCH_DIR" ]]; then
                print_error "未找到LibTorch，请使用 -l 选项指定路径"
                print_info "下载LibTorch: https://pytorch.org/get-started/locally/"
                exit 1
            fi
        else
            if [[ ! -d "$LIBTORCH_DIR" ]]; then
                print_error "LibTorch目录不存在: $LIBTORCH_DIR"
                exit 1
            fi
            
            if [[ ! -f "$LIBTORCH_DIR/share/cmake/Torch/TorchConfig.cmake" ]]; then
                print_error "LibTorch配置文件不存在: $LIBTORCH_DIR/share/cmake/Torch/TorchConfig.cmake"
                exit 1
            fi
        fi
    fi
    
    # 检查CUDA
    if [[ "$USE_CUDA" == "ON" ]]; then
        if ! command -v nvcc &> /dev/null; then
            print_warning "NVCC未找到，CUDA支持可能不可用"
        else
            print_info "CUDA版本: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
        fi
    fi
    
    print_message "依赖检查完成"
}

# 清理构建目录
clean_build() {
    if [[ "$CLEAN_BUILD" == "ON" ]] && [[ -d "$BUILD_DIR" ]]; then
        print_message "清理构建目录: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
}

# 创建构建目录
create_build_dir() {
    print_message "创建构建目录: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
}

# 配置CMake
configure_cmake() {
    print_message "配置CMake..."
    
    cd "$BUILD_DIR"
    
    CMAKE_ARGS=(
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DUSE_LIBTORCH=$USE_LIBTORCH"
        "-DUSE_CUDA=$USE_CUDA"
        "-DUSE_OPENCV=$USE_OPENCV"
    )
    
    if [[ "$USE_LIBTORCH" == "ON" && -n "$LIBTORCH_DIR" ]]; then
        CMAKE_ARGS+=("-DLibTorch_DIR=$LIBTORCH_DIR")
    fi
    
    if [[ "$VERBOSE" == "ON" ]]; then
        CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
    fi
    
    print_info "CMake参数: ${CMAKE_ARGS[*]}"
    
    cmake "${CMAKE_ARGS[@]}" ..
    
    if [[ $? -ne 0 ]]; then
        print_error "CMake配置失败"
        exit 1
    fi
    
    cd ..
}

# 构建项目
build_project() {
    print_message "构建项目..."
    
    cd "$BUILD_DIR"
    
    # 确定并行任务数
    if command -v nproc &> /dev/null; then
        JOBS=$(nproc)
    else
        JOBS=4
    fi
    
    MAKE_ARGS=("-j$JOBS")
    
    if [[ "$VERBOSE" == "ON" ]]; then
        MAKE_ARGS+=("VERBOSE=1")
    fi
    
    print_info "并行任务数: $JOBS"
    
    make "${MAKE_ARGS[@]}"
    
    if [[ $? -ne 0 ]]; then
        print_error "构建失败"
        exit 1
    fi
    
    cd ..
}

# 运行测试
run_tests() {
    print_message "运行测试..."
    
    cd "$BUILD_DIR"
    
    if [[ -f "test_dsnet" ]]; then
        print_info "运行基本测试..."
        ./test_dsnet
        
        if [[ $? -ne 0 ]]; then
            print_warning "测试失败"
        else
            print_message "测试通过"
        fi
    else
        print_warning "测试可执行文件不存在"
    fi
    
    cd ..
}

# 显示构建结果
show_results() {
    print_message "构建完成!"
    
    print_info "生成的文件:"
    if [[ -f "$BUILD_DIR/libdsnet_inference.so" ]]; then
        print_info "  动态库: $BUILD_DIR/libdsnet_inference.so"
    fi
    
    if [[ -f "$BUILD_DIR/libdsnet_inference.a" ]]; then
        print_info "  静态库: $BUILD_DIR/libdsnet_inference.a"
    fi
    
    if [[ -f "$BUILD_DIR/inference_example" ]]; then
        print_info "  示例程序: $BUILD_DIR/inference_example"
    fi
    
    if [[ -f "$BUILD_DIR/test_dsnet" ]]; then
        print_info "  测试程序: $BUILD_DIR/test_dsnet"
    fi
    
    print_info ""
    print_info "使用方法:"
    print_info "  1. 转换PyTorch模型: python convert_model_to_torchscript.py"
    print_info "  2. 运行示例: ./$BUILD_DIR/inference_example"
    print_info "  3. 运行测试: ./$BUILD_DIR/test_dsnet"
}

# 主执行流程
main() {
    check_dependencies
    clean_build
    create_build_dir
    configure_cmake
    build_project
    run_tests
    show_results
}

# 错误处理
trap 'print_error "构建过程中发生错误，退出代码: $?"' ERR

# 执行主流程
main

print_message "构建脚本执行完成"
