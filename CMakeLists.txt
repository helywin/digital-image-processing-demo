cmake_minimum_required(VERSION 3.17)
project(opencv_demo)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/out)

find_package(OpenCV 4.4 REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_demo main.cpp)
target_link_libraries(opencv_demo ${OpenCV_LIBS})

add_executable(histogram histogram.cpp)
target_link_libraries(histogram ${OpenCV_LIBS})

add_executable(simple_smooth simple_smooth.cpp)
target_link_libraries(simple_smooth ${OpenCV_LIBS})

add_executable(simple_sharpening simple_sharpening.cpp)
target_link_libraries(simple_sharpening ${OpenCV_LIBS})

add_executable(dft dft.cpp)
target_link_libraries(dft ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable(butterworth butterworth.cpp)
target_link_libraries(butterworth ${OpenCV_LIBS})

file(COPY ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json DESTINATION ${CMAKE_SOURCE_DIR})
