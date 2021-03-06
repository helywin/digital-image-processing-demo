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

add_executable(spectrum_filter spectrum_filter.cpp dftshift.cpp dftshift.hpp)
target_link_libraries(spectrum_filter ${OpenCV_LIBS})

add_executable(space_filtering space_filtering.cpp dftshift.cpp dftshift.hpp)
target_link_libraries(space_filtering ${OpenCV_LIBS})

add_executable(turbulance turbulance.cpp dftshift.cpp dftshift.hpp)
target_link_libraries(turbulance ${OpenCV_LIBS})

#add_executable(wiener_filtering wiener_filtering.cpp)
#target_link_libraries(wiener_filtering ${OpenCV_LIBS})

add_executable(wavelet_transform wavelet_transform.cpp)
target_link_libraries(wavelet_transform ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

set(COMPILES_COMMANDS ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json)
if (EXISTS ${COMPILES_COMMANDS})
    file(COPY ${COMPILES_COMMANDS} DESTINATION ${CMAKE_SOURCE_DIR})
endif ()
