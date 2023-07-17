cmake_minimum_required(VERSION 3.8)
project(industrial_camera)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-dev)


find_package(cv_bridge REQUIRED)
find_package(foxglove_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library( mvsdk SHARED IMPORTED )
#set_target_properties( mvsdk PROPERTIES IMPORTED_LOCATION lib/libMVSDK.so )
set_target_properties( mvsdk PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/arm64/libMVSDK.so")

add_executable( industrial_camera src/main.cpp  src/camera.cpp)
target_include_directories(industrial_camera PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

ament_target_dependencies(
  industrial_camera
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


target_link_libraries( industrial_camera mvsdk )

install(TARGETS industrial_camera
   DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
