cmake_minimum_required(VERSION 3.8)
project(lock_free_queue)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-dev)

add_library(queue_lib SHARED src/lock_free_queue.cpp)
target_include_directories(queue_lib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

install(TARGETS queue_lib LIBRARY DESTINATION lib)

if(BUILD_TESTING)
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo "start")
    add_subdirectory("tests")
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo "stop")
endif()
