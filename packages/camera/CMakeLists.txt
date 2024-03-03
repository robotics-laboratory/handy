cmake_minimum_required(VERSION 3.8)
project(camera)

add_compile_options(-Wall -Wextra -Werror=unused-variable -Wpedantic)

find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(foxglove_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(camera_srvs REQUIRED)

add_library(mvsdk SHARED IMPORTED)
set_target_properties(mvsdk PROPERTIES IMPORTED_LOCATION "/lib/libMVSDK.so")

add_executable(camera src/camera_main.cpp  src/camera.cpp src/params.cpp src/camera_status.cpp)
add_executable(calibration src/calibration_main.cpp  src/calibration.cpp src/params.cpp)

target_include_directories(camera PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "/usr/include")
target_include_directories(calibration PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "/usr/include")

ament_target_dependencies(
    camera
    rclcpp
    cv_bridge
    OpenCV
    sensor_msgs
    yaml_cpp_vendor)

ament_target_dependencies(
  calibration
    rclcpp
    cv_bridge
    OpenCV
    sensor_msgs
    yaml_cpp_vendor
    std_srvs
    foxglove_msgs
    visualization_msgs
    geometry_msgs
    std_msgs
    camera_srvs)

target_link_libraries(camera mvsdk)

install(TARGETS camera calibration
   DESTINATION lib/${PROJECT_NAME})

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/params")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/params")
endif()

install(DIRECTORY
    launch params
    DESTINATION share/${PROJECT_NAME})

ament_package()
