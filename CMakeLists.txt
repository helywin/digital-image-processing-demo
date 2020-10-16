cmake_minimum_required(VERSION 3.17)
project(opencv_demo)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/out)

find_package(OpenCV 4.4 REQUIRED)

add_executable(opencv_demo main.cpp)
target_link_libraries(opencv_demo ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(histogram histogram.cpp)
target_link_libraries(histogram ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})
