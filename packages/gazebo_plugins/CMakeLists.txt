cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Find Gazebo
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Build our plugin
add_library(BallAerodynamicsPlugin SHARED src/ball_aerodynamics.cpp include/ball_aerodynamics.hh)
target_include_directories(BallAerodynamicsPlugin PUBLIC include)
target_link_libraries(BallAerodynamicsPlugin ${GAZEBO_LIBRARIES})