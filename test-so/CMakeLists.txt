cmake_minimum_required(VERSION 2.8)

project(yolo_test)

add_definitions(-DOPENCV)
set( CMAKE_CXX_FLAGS "-std=c++11" )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_library(darknet libdarknet.so /home/xzw/WorkSpace/darknet-so/)

include_directories(/home/xzw/WorkSpace/darknet-so/include)

add_executable(yolo_test test.cpp)
target_link_libraries(yolo_test ${OpenCV_LIBS} ${darknet})
