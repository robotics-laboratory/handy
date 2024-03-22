cmake_minimum_required(VERSION 3.8)
project(camera_identifier)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-dev)

add_library(mvsdk SHARED IMPORTED)
set_target_properties(mvsdk PROPERTIES IMPORTED_LOCATION "/lib/libMVSDK.so")

add_executable(camera_identifier src/identifier.cpp
                                 ../camera/src/camera_status.cpp)
target_include_directories(
  camera_identifier
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../camera/include>"
         "/usr/include")

target_link_libraries(camera_identifier mvsdk)

install(TARGETS camera_identifier DESTINATION lib/${PROJECT_NAME})
