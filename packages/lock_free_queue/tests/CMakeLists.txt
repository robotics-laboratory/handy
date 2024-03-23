find_package(ament_cmake_gtest REQUIRED)

ament_add_gtest(queue_test ../src/lock_free_queue.cpp src/test.cpp)

# ament_target_dependencies(
#     camera_test
#     OpenCV
#     yaml_cpp_vendor)

target_include_directories(queue_test PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
  "/usr/include")