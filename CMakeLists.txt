cmake_minimum_required(VERSION 2.8)
project(opticalFlow)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Farneback Farneback.cpp)
add_executable(LucasKanade LucasKanade.cpp)
add_executable(simpleFlow simpleFlow.cpp)
target_Link_Libraries(Farneback ${OpenCV_LIBS})
target_Link_Libraries(LucasKanade ${OpenCV_LIBS})
target_Link_Libraries(simpleFlow ${OpenCV_LIBS})