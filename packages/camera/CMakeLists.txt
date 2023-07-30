cmake_minimum_required(VERSION 3.8)
project(camera)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-dev)

find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)

add_library( mvsdk SHARED IMPORTED )
set_target_properties( mvsdk PROPERTIES IMPORTED_LOCATION "/lib/libMVSDK.so")

add_executable( camera src/main.cpp  src/camera.cpp)
#add_executable( calibration_node src/calib_main.cpp  src/calibration.cpp)

target_include_directories(camera PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "/usr/include")

ament_target_dependencies(
    camera
    rclcpp
    cv_bridge
    OpenCV
    sensor_msgs
    yaml_cpp_vendor)

target_link_libraries(camera mvsdk)

install(TARGETS camera
   DESTINATION lib/${PROJECT_NAME})
  
install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
