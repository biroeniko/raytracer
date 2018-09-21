cmake_minimum_required(VERSION 2.8)

project(raytracer)

include_directories(	include
 			include/util)

file(GLOB SOURCES
    src/*.cpp
    src/util/*.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
