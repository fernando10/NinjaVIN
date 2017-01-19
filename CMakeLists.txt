project(NinjaVIN)
cmake_minimum_required(VERSION 2.8)


# Mac ONLY. Define Relative Path on Mac OS
if(NOT DEFINED CMAKE_MACOSX_RPATH)
  set(CMAKE_MACOSX_RPATH 0)
endif()


set( CMAKE_CXX_FLAGS "-std=c++0x -Wall ${CMAKE_CXX_FLAGS}" )
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release )
endif()

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# require OpenCV
find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS})

# require okvis
find_package( okvis 1.1 REQUIRED)
include_directories(${OKVIS_INCLUDE_DIRS})

# require brisk
find_package( brisk 2 REQUIRED)
include_directories(${BRISK_INCLUDE_DIRS})

# require ceres
list(APPEND CMAKE_PREFIX_PATH ${OKVIS_INSTALLATION})
find_package( Ceres REQUIRED )
include_directories(${CERES_INCLUDE_DIRS})

# require OpenGV
find_package(opengv REQUIRED)

# require HAL
find_package(HAL REQUIRED)
include_directories(${HAL_INCLUDE_DIRS})

set(HRDS_LIST
    InterpolationBuffer.h)

aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST} ${HDRS_LIST})

target_link_libraries( ${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${HAL_LIBRARIES}
  ${OKVIS_LIBRARIES}
  ${BRISK_LIBRARIES}
  ${CERES_LIBRARIES}
  )
