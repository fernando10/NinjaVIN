project(VINSystem)
cmake_minimum_required(VERSION 2.8)


# Mac ONLY. Define Relative Path on Mac OS
if(NOT DEFINED CMAKE_MACOSX_RPATH)
  set(CMAKE_MACOSX_RPATH 0)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules/")

string(TOLOWER ${PROJECT_NAME} LIBRARY_NAME)

include(install_package)
include(SetPlatformVars)


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

option(BUILD_EXAMPLE "Build Example" ON)


#####################################################################
## Find required packages
enable_testing()
find_package(Eigen3 REQUIRED)
find_package(GLog REQUIRED)
find_package(sdtrack REQUIRED)
find_package(BA REQUIRED)
find_package(HAL REQUIRED)
find_package(Sophus REQUIRED)
find_package(Protobuf REQUIRED)
find_package( OpenCV REQUIRED core calib3d )


set( REQUIRED_INCLUDE_DIRS
    ${GLOG_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${sdtrack_INCLUDE_DIRS}
    ${HAL_INCLUDE_DIRS}
    ${BA_INCLUDE_DIRS}
   )

set( REQUIRED_LIBRARIES
    ${GLOG_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${HAL_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    ${BA_LIBRARIES}
    ${sdtrack_LIBRARIES}
    )

set( INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME} )

set( HEADERS
    ${INC_DIR}/VINSystem.hpp
    )

include_directories( ${INC_DIR}/common )


set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src )

set( SOURCES
    ${SRC_DIR}/VINSystem.cpp
    )


include_directories( ${REQUIRED_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include )
add_library( ${LIBRARY_NAME} SHARED ${SOURCES} ${HEADERS} )
target_link_libraries( ${LIBRARY_NAME} ${REQUIRED_LIBRARIES} )
set_target_properties( ${LIBRARY_NAME} PROPERTIES VERSION "${PACKAGE_VERSION}" SOVERSION "${PACKAGE_VERSION}" )

install_package(
    PKG_NAME ${PROJECT_NAME}
    LIB_NAME ${LIBRARY_NAME}
    VERSION ${PACKAGE_VERSION}
    DESCRIPTION "ARPG SLAM Library (sdtrack)"
    INSTALL_INCLUDE_DIR true
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    INCLUDE_DIRS ${REQUIRED_INCLUDE_DIRS}
    LINK_LIBS ${REQUIRED_LIBRARIES}
    )


add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

# Build application
if(BUILD_EXAMPLE)
  set( Compass_DIR ${CMAKE_CURRENT_BINARY_DIR} )
  add_subdirectory(example)
endif()
