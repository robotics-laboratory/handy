cmake_minimum_required(VERSION 2.8)
project( demo )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ../../include)

add_library( mvsdk SHARED IMPORTED )
#set_target_properties( mvsdk PROPERTIES IMPORTED_LOCATION lib/libMVSDK.so )
set_target_properties( mvsdk PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/arm64/libMVSDK.so")

add_executable( demo src/main.cpp )
target_include_directories(demo PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

target_link_libraries( demo ${OpenCV_LIBS} mvsdk )
