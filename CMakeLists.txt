cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)

project(raytracer)

include_directories(	include
 			include/util
			include/hitables
			include/examples
			include/materials
)

file(GLOB SOURCES
    src/*.cpp
    src/util/*.cpp
    src/hitables/*.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
