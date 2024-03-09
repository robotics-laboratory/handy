find_package(ament_cmake_gtest REQUIRED)

ament_add_gtest(camera_test test.cpp ../src/params.cpp)

ament_target_dependencies(
    camera_test
    OpenCV
    yaml_cpp_vendor)

target_include_directories(camera_test PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
  "/usr/include")
