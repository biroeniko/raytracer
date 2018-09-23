cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)

project(raytracer)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# header files
include_directories(	include
 			include/util
			include/hitables
			include/examples
			include/materials
)

# source files
file(GLOB SOURCES
    src/*.cpp
    src/util/*.cpp
    src/hitables/*.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
