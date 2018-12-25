cmake_minimum_required(VERSION 2.8)

project(raytracer)

# Compiler
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -DNDEBUG -mavx2")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-std=c++11 -g ${CMAKE_CXX_FLAGS_RELEASE}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# SDL2
find_package(SDL2 REQUIRED)
string(STRIP ${SDL2_LIBRARIES} SDL2_LIBRARIES)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# CUDA
option(CUDA_SUPPORT "CUDA support" ON)
if(CUDA_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_SUPPORT -DPRIME_ATOM32_GCC")
    find_package(CUDA REQUIRED)
    set(CUDA_HOST_COMPILER gcc)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G -std=c++11 -O3 -DNDEBUG -use_fast_math -Xptxas -v -keep -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70")
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if(CUDA_SUPPORT)
    add_definitions(-DCUDA_ENABLED)
endif()

# Header files
include_directories(
    ${PROJECT_SOURCE_DIR}/src 
    ${SDL2_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/stb_image
)

# Source files

set(SRC_COMMON

)

set(SRC_HOST
    ${SRC_COMMON}
    src/hitables/hitable.h
    src/hitables/hitableList.h
    src/hitables/sphere.h
    src/materials/material.h
    src/util/vec3.h
    src/util/ray.h
    src/util/image.h
    src/util/scene.h
    src/util/window.h
    src/util/renderer.h
    src/util/renderer.cpp
    src/util/camera.h
    src/util/camera.cpp
    src/util/randomGenerator.h
    src/util/util.h
    src/util/util.cpp
    src/util/common.h
    src/raytracer.cpp
)

if (CUDA_SUPPORT)
    set(SRC_HOST
        ${SRC_HOST}
        src/util/image.cu
        src/util/renderer.cu
        src/util/scene.cu
        src/raytracer.cu
    )
endif()

if (CUDA_SUPPORT)
    set_source_files_properties(src/util/image.cu
                                PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
endif()

if (CUDA_SUPPORT)
    cuda_add_executable(${PROJECT_NAME} ${SRC_HOST} OPTIONS -x cu)
else()
    add_executable(${PROJECT_NAME} ${SRC_HOST})
endif()

target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES} -lcudart)
