cmake_minimum_required(VERSION 3.8)
project(camera)

add_compile_options(-Wall -Wextra -Werror=unused-variable -Wpedantic)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3")

set(Torch_DIR "/usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(OpenCV REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED)
find_package(mcap_vendor REQUIRED)

add_library(mvsdk SHARED IMPORTED)
set_target_properties(mvsdk PROPERTIES IMPORTED_LOCATION "/lib/libMVSDK.so")

add_executable(camera src/camera_main.cpp src/camera.cpp src/camera_status.cpp)
add_executable(
  calibration src/calibration_main.cpp src/calibration.cpp src/params.cpp
)
add_executable(triangulation src/triangulation.cpp src/params.cpp)
add_executable(inference_node src/inference_node.cpp)

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
target_include_directories(
  inference_node PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                       "/usr/include"
)

ament_target_dependencies(camera yaml_cpp_vendor mcap_vendor)
ament_target_dependencies(calibration yaml_cpp_vendor Boost OpenCV)
ament_target_dependencies(triangulation OpenCV yaml_cpp_vendor)
ament_target_dependencies(inference_node OpenCV Boost Torch yaml_cpp_vendor mcap_vendor)

target_link_libraries(camera mvsdk)
target_link_libraries(inference_node mvsdk "${TORCH_LIBRARIES}")

install(TARGETS camera calibration triangulation inference_node
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
