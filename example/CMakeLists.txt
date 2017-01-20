find_package(VINSystem REQUIRED)

set( REQUIRED_LIBRARIES
    ${VINSystem_LIBRARIES}
    )

set( REQUIRED_INCLUDE_DIRS
    ${VINSystem_INCLUDE_DIRS}
    )

#####################################################################
## Find optional packages
find_package(SLAMViewer QUIET)

if(SLAMViewer_FOUND)
  add_definitions(-DHAVE_SLAMViewer)
  list(APPEND REQUIRED_LIBRARIES ${SLAMViewer_LIBRARIES})
  list(APPEND REQUIRED_INCLUDE_DIRS ${SLAMViewer_INCLUDE_DIRS})
endif()
#####################################################################

include_directories(
    ${REQUIRED_INCLUDE_DIRS}
    )

set(SRCS
    example_vin.cpp
   )

add_executable(example_vin ${SRCS})

target_link_libraries( example_vin
  ${REQUIRED_LIBRARIES}
  )
