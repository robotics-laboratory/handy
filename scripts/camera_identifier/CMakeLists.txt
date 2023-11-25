cmake_minimum_required(VERSION 3.8)
project(camera_identifier)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-dev)

add_library(mvsdk SHARED IMPORTED)
set_target_properties(mvsdk PROPERTIES IMPORTED_LOCATION "/lib/libMVSDK.so")

add_executable(camera_identifier src/identifier.cpp)
target_link_libraries(camera_identifier mvsdk)
