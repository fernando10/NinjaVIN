find_package(VINSystem REQUIRED)

include_directories(
    ${VINSystem_INCLUDE_DIRS}
    )

set(SRCS
    example_vin.cpp
   )

add_executable(example_vin ${SRCS})

target_link_libraries( example_vin
  ${VINSystem_LIBRARIES}
  )
