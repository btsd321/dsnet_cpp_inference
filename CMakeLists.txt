cmake_minimum_required(VERSION 3.16)
project(dsnet_cpp_inference VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 如果有vcpkg，使用vcpkg工具链
if(EXISTS "~/SoftWare/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "~./SoftWare/vcpkg/scripts/buildsystems/vcpkg.cmake")
    message(STATUS "Using vcpkg toolchain")
endif()

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 查找依赖包
find_package(Eigen3 REQUIRED)
find_package(OpenCV QUIET)

# argparse是header-only库，直接包含vcpkg安装的路径
if(EXISTS "~/SoftWare/vcpkg/installed/x64-linux/include/argparse")
    set(ARGPARSE_INCLUDE_DIR "~/SoftWare/vcpkg/installed/x64-linux/include")
    message(STATUS "Found argparse in vcpkg")
endif()

# 可选：查找CUDA（如果需要GPU推理）
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
else()
    message(STATUS "CUDA not found, using CPU only")
endif()

# 可选：查找LibTorch（如果需要PyTorch C++推理）
find_package(Torch QUIET)
if(Torch_FOUND)
    add_definitions(-DUSE_LIBTORCH)
    message(STATUS "LibTorch found")
else()
    message(STATUS "LibTorch not found, using custom inference")
endif()

# 添加子目录
add_subdirectory(thirdparty)

# 创建dsnet推理库
set(DSNET_SOURCES
    src/dsnet_inference.cpp
)

set(DSNET_HEADERS
    include/dsnet_inference.h
)

# 创建静态库
add_library(dsnet_inference STATIC ${DSNET_SOURCES})

# 链接库
target_link_libraries(dsnet_inference
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

if(Torch_FOUND)
    target_link_libraries(dsnet_inference ${TORCH_LIBRARIES})
endif()

if(CUDA_FOUND)
    target_link_libraries(dsnet_inference ${CUDA_LIBRARIES})
endif()

# 设置库的包含目录
target_include_directories(dsnet_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# 创建测试可执行文件
add_executable(dsnet_test src/test.cpp)
target_include_directories(dsnet_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${ARGPARSE_INCLUDE_DIR}
)
target_link_libraries(dsnet_test dsnet_inference)

# 安装规则
install(TARGETS dsnet_inference
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${DSNET_HEADERS}
    DESTINATION include
)

install(TARGETS dsnet_test
    DESTINATION bin
)

# 添加测试
enable_testing()
add_test(NAME dsnet_basic_test COMMAND dsnet_test)

# 打印配置信息
message(STATUS "")
message(STATUS "=================== DSNet C++ Inference Configuration ===================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 Version: ${Eigen3_VERSION}")
if(CUDA_FOUND)
    message(STATUS "CUDA Version: ${CUDA_VERSION}")
endif()
if(Torch_FOUND)
    message(STATUS "LibTorch Version: ${Torch_VERSION}")
endif()
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================================================")
message(STATUS "")