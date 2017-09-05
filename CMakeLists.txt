project(KalmanFilterController)

cmake_minimum_required(VERSION 2.8.7)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
set(CMAKE_C_COMPILER, "/usr/bin/clang")
set(CMAKE_CXX_COMPILER, "/usr/bin/clang++")

#add_subdirectory(main)
add_subdirectory(Test)
add_subdirectory(Lib)
