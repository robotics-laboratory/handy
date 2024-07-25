cmake_minimum_required(VERSION 3.8)
project(camera)

add_compile_options(-Wall -Wextra -Werror=unused-variable -Wpedantic)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)

find_package(OpenCV REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED)
find_package(mcap REQUIRED)
# find_package(LZ4 REQUIRED)

add_library(mvsdk SHARED IMPORTED)
set_target_properties(mvsdk PROPERTIES IMPORTED_LOCATION "/lib/libMVSDK.so")

add_executable(camera src/camera_main.cpp src/camera.cpp src/camera_status.cpp)
add_executable(
  calibration src/calibration_main.cpp src/calibration.cpp src/params.cpp
)
add_executable(triangulation src/triangulation.cpp src/params.cpp)

target_include_directories(
  camera PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>" "/usr/include"
)

target_include_directories(
  calibration PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                     "/usr/include"
)
target_include_directories(
  triangulation PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                       "/usr/include"
)

ament_target_dependencies(camera yaml_cpp_vendor)
ament_target_dependencies(calibration yaml_cpp_vendor Boost OpenCV)
ament_target_dependencies(triangulation OpenCV yaml_cpp_vendor)

target_link_libraries(camera mvsdk mcap)

install(TARGETS camera calibration triangulation
        DESTINATION lib/${PROJECT_NAME}
)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/params")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/params")
endif()

install(DIRECTORY launch params DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  add_subdirectory("test")
endif()

ament_package()
