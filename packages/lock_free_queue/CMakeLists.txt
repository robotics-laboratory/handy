cmake_minimum_required(VERSION 3.8)
project(lock_free_queue)

find_package(ament_cmake_gtest REQUIRED)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-dev)

add_library(queue_lib SHARED src/lock_free_queue.cpp)
target_include_directories(queue_lib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

install(TARGETS queue_lib LIBRARY DESTINATION lib)

ament_add_gtest(queue_test tests/src/test.cpp src/lock_free_queue.cpp)
target_include_directories(queue_test PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

ament_package()
