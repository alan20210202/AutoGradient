# CMakeList.txt: AutoGradient 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 将源代码添加到此项目的可执行文件。

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenMP)
file(GLOB SOURCES "./*.cpp")
add_executable (AutoGradient ${SOURCES})
target_link_libraries(AutoGradient PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
	target_link_libraries(AutoGradient PUBLIC OpenMP::OpenMP_CXX)	
endif()
# TODO: 如有需要，请添加测试并安装目标。
