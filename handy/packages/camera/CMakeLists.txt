cmake_minimum_required(VERSION 3.8)
project(camera)

add_compile_options(-Wall -Wextra -Wpedantic)

# find dependencies
find_package(cv_bridge REQUIRED)
find_package(foxglove_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)


add_executable(camera_node src/camera_node.cpp)
#add_executable(moving_point_client src/moving_point_client.cpp)


target_include_directories(camera_node PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

# target_include_directories(moving_point_client PUBLIC
#   "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
#   "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

ament_target_dependencies(
  camera_node
    OpenCV
    rclcpp
    cv_bridge
    foxglove_msgs
    OpenCV
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    visualization_msgs)
    


install(TARGETS camera_node
   DESTINATION lib/${PROJECT_NAME})

# install(
#     DIRECTORY launch config
#     DESTINATION share/${PROJECT_NAME})

ament_package()