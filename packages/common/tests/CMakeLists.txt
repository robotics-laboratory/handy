ament_add_gtest(queue_test src/test.cpp)
target_include_directories(
  queue_test PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(queue_test ${PROJECT_NAME})
