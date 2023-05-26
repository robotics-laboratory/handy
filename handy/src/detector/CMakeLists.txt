cmake_minimum_required(VERSION 3.8)
project(detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(Torch_DIR "/usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(realsense2 REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(Torch REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 ${TORCH_CXX_FLAGS}")


add_executable(detector src/detector.cpp)
ament_target_dependencies(detector visualization_msgs rclcpp sensor_msgs OpenCV Torch cv_bridge realsense2 tf2_msgs)
target_link_libraries(detector "${TORCH_LIBRARIES}")


install(TARGETS
  detector
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
