# ThirdParty Dependencies
cmake_minimum_required(VERSION 3.16)

# 创建一个空的thirdparty目标，以防有第三方库需要编译
add_custom_target(thirdparty_dependencies)

# 这里可以添加第三方依赖的编译规则
# 例如：
# add_subdirectory(some_third_party_lib)

message(STATUS "ThirdParty dependencies configured")
