cmake_minimum_required(VERSION 3.8)

project(camera_srvs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME} "srv/CalibrationCommand.srv"
                           DEPENDENCIES builtin_interfaces)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
