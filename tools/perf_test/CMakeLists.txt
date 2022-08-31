# Copyright (c) intellif Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(perf_test)

# cmake set info
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if ("${RUN_TYPE}" STREQUAL "FULLHAN")
    set(CMAKE_CXX_COMPILER $ENV{ARM_CXX_COMPILER})
    set(CMAKE_C_COMPILER $ENV{ARM_C_COMPILER})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../output")

set(COMMON_COMPILER_FLAG "-Wall -fPIC -fvisibility=hidden")

if ("${RUN_TYPE}" STREQUAL "FULLHAN")
    set(COMMON_COMPILER_FLAG "${COMMON_COMPILER_FLAG} -mfloat-abi=hard -mfpu=neon-vfpv4 -mcpu=cortex-a7 -D_GNU_SOURCE")
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-O0 -g -std=c++11 ${COMMON_COMPILER_FLAG} ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-O2 -std=c++11 ${COMMON_COMPILER_FLAG} ${CMAKE_CXX_FLAGS}")
endif()

set(SDK_INC_PATH "$ENV{DENGINE_ROOT}/tyhcp/include")

if ("${RUN_TYPE}" STREQUAL "FULLHAN")
    set(SDK_LIB_PATH "$ENV{DENGINE_ROOT}/tyhcp/platform/$ENV{CROSS_PLAT}/lib/dynamic")
    set(3RD_LIB_PATH "$ENV{DENGINE_ROOT}/tyexamples/3rdparty/fullhan/$ENV{CROSS_PLAT}/lib/dynamic")
    add_definitions(-DDP2000)
else()
    set(SDK_LIB_PATH "$ENV{DENGINE_ROOT}/tyhcp/platform/$ENV{HOST_PLAT}/lib/dynamic")
    add_definitions(-DCPU_SIMU)
endif()

set(INFER_INC_PATH "src/common")
set(EG_INC_PATH "src")
MESSAGE (${SDK_INC_PATH})
MESSAGE (${INFER_INC_PATH})

# Header path
include_directories(  
    ${SDK_INC_PATH}
    ${INFER_INC_PATH}
    ${EG_INC_PATH}
)

# add host lib path
link_directories(
    ${SDK_LIB_PATH}
    ${3RD_LIB_PATH}
)

add_executable(${CMAKE_PROJECT_NAME}
    src/common/utils.cpp
    src/common/random.cpp
    src/common/model_process.cpp
    src/common/device_process.cpp
    src/main.cpp)

# desdk libs
list(APPEND SDK_LIBS dcl_memory)
list(APPEND SDK_LIBS dcl_runtime)
list(APPEND SDK_LIBS dcl_utils)
list(APPEND SDK_LIBS stream_sched)
list(APPEND SDK_LIBS mdl_parse)
list(APPEND SDK_LIBS drv_dp2000)
list(APPEND SDK_LIBS dcl_ioctl)
list(APPEND SDK_LIBS dcl_ioctl)
if ("${RUN_TYPE}" STREQUAL "FULLHAN")
    list(APPEND SDK_LIBS mpi)
    list(APPEND SDK_LIBS dsp)
endif()


#opencv libs
list(APPEND OPENCV_LIBS opencv_core)
list(APPEND OPENCV_LIBS opencv_imgproc)
list(APPEND OPENCV_LIBS opencv_highgui)
list(APPEND OPENCV_LIBS opencv_imgcodecs)

if ("${RUN_TYPE}" STREQUAL "FULLHAN")
    target_link_libraries(${CMAKE_PROJECT_NAME} ${SDK_LIBS} vmm pthread dl)
else()
    target_link_libraries(${CMAKE_PROJECT_NAME} ${SDK_LIBS} ${OPENCV_LIBS} pthread dl)
endif()
