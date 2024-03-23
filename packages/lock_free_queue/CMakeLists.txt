cmake_minimum_required(VERSION 3.8)
project(lock_free_queue)

find_package(ament_cmake_gtest REQUIRED)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-dev)

add_library(queue_lib INTERFACE)
target_include_directories(queue_lib INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

ament_add_gtest(queue_test tests/src/test.cpp)
target_include_directories(queue_test PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(queue_test queue_lib)

ament_package()
