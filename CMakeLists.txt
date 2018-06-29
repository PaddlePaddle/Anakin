# Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(cmake/thirdparty_version.cmake)
project(ANAKIN C CXX)
include(cmake/msg_color.cmake)
include(cmake/utils.cmake)
include(cmake/statistic.cmake)

# ----------------------------------------------------------------------------
# section: global anakin version and lib name
# ----------------------------------------------------------------------------
# global anakin version 0.1.0
set(VERSION_MAJOR "0")
set(VERSION_MINOR "1")
set(VERSION_PATCH "0")
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# anakin lib name and global directories
set(anakin_lib_so "anakin")
set(anakin_lib_static "anakin_static")
# set root dir of modules
set(ANAKIN_ROOT ${PROJECT_SOURCE_DIR})
include_directories(${ANAKIN_ROOT})

set(ANAKIN_FRAMEWORK ${ANAKIN_ROOT}/framework)
set(ANAKIN_LITE ${ANAKIN_FRAMEWORK}/lite)
set(ANAKIN_UTILS ${ANAKIN_ROOT}/utils)
set(ANAKIN_THIRD_PARTY_PATH ${ANAKIN_ROOT}/third-party)
set(ANAKIN_MODEL_PARSER ${ANAKIN_FRAMEWORK}/model_parser)
set(ANAKIN_SABER ${ANAKIN_ROOT}/saber)
set(ANAKIN_UNIT_TEST ${ANAKIN_ROOT}/test)
set(ANAKIN_EXAMPLES ${ANAKIN_ROOT}/examples)


# ----------------------------------------------------------------------------
# section: options for anakin
# ----------------------------------------------------------------------------
#anakin data float precision
anakin_option(ANAKIN_TYPE_FP64 "define the FP64 for data precision." NO)
anakin_option(ANAKIN_TYPE_FP32 "define the FP32 for data precision." YES)
anakin_option(ANAKIN_TYPE_FP16 "define the FP16 for data precision." NO)
anakin_option(ANAKIN_TYPE_INT8 "define the INT8 for data precision." NO)

#select the plantform to build
anakin_option(USE_GPU_PLACE "Select the build mode for GPU place." YES)
anakin_option(USE_X86_PLACE "Select the build mode for X86 place." NO)
anakin_option(USE_ARM_PLACE "Select the build mode for ARM place." NO)

# plantfrom details
anakin_option(NVIDIA_GPU "Use NVIDIA GPU place." YES if USE_GPU_PLACE)
anakin_option(AMD_GPU "Use AMD GPU place." NO if USE_GPU_PLACE AND NOT NVIDIA_GPU)
anakin_option(TARGET_ANDROID "" YES if USE_ARM_PLACE)
anakin_option(TARGET_IOS "" NO if USE_ARM_PLACE)

# compile options for NVIDIA_GPU place
anakin_option(USE_CUDA "Use Cuda libs." YES if NVIDIA_GPU)
anakin_option(USE_CUBLAS "Use Cublas libs." YES if USE_CUDA)
anakin_option(USE_CURAND "Use Curand libs." YES if USE_CUDA)
anakin_option(USE_CUFFT "Use CuFFT libs." YES if USE_CUDA)
anakin_option(USE_CUDNN "Use Cudnn libs." YES if USE_CUDA)
anakin_option(BUILD_CROSS_PLANTFORM "Build anakin lib for any nvidia device plantform." YES if USE_CUDA)
anakin_option(BUILD_FAT_BIN "Build anakin cuda fat-bin lib for all device plantform" NO if BUILD_CROSS_PLANTFORM)

cmake_minimum_required(VERSION ${MIN_CMAKE_V} FATAL_ERROR)

if(USE_CUDA)
    # Select gpu target arch for local high performance implement sass code . Now we have checked on sm_61 sm_50 and it works well.
    set(SELECTED_SASS_TARGET_ARCH "61")
endif()
if((NOT BUILD_FAT_BIN) AND (NOT BUILD_CROSS_PLANTFORM) AND USE_CUDA)
    # Select the only nvidia gpu arch you want to be built on
    set(TARGET_GPUARCH 6.1)
endif()

# build options for cuda.
anakin_option(BUILD_CUBIN "BUILD with the -cubin option in Device mode" NO if USE_CUDA)
anakin_option(COMPILE_PTX "Returns a list of PTX files generated from src." NO if USE_CUDA)


# common build options
anakin_option(ENABLE_DEBUG "Enable DEBUG(default) mode." YES)
anakin_option(ENABLE_VERBOSE_MSG "Enable verbose=1 : compile msg during make." NO)
anakin_option(DISABLE_ALL_WARNINGS "Disable all the warning msg during compile." YES)
anakin_option(ENABLE_NOISY_WARNINGS "Enable noisy warning msg during compile." NO if DISABLE_ALL_WARNINGS)

# using 3rd party libs 
anakin_option(USE_LOGGER "Build native logger components." YES)
anakin_option(USE_GLOG "Build Glog components." NO if NOT USE_LOGGER)
anakin_option(USE_PROTOBUF "Build Google protobuf components." YES)
anakin_option(USE_OPENCV "Use static opencv libs." NO)
anakin_option(USE_BOOST "Use static BOOST libs." NO)
anakin_option(USE_OPENMP "Use Openmp when in android environment." YES if TARGET_ANDROID)
anakin_option(USE_GTEST "Use googletest libs." NO if BUILD_WITH_UNIT_TEST)
anakin_option(USE_PYTHON "Generate py wrappers." NO)
anakin_option(USE_OPENCL "Use OpenCL ." NO)
anakin_option(USE_GFLAGS "Build Google gflags components." NO)
anakin_option(USE_MKL "Use mkl libs." NO if USE_X86_PLACE)
anakin_option(USE_MKLML "Use MKLML libs." YES if USE_X86_PLACE)
anakin_option(USE_XBYAK "Use XBYAK libs." YES if USE_X86_PLACE)
anakin_option(USE_OPENMP "Use Openmp when in android environment." YES if TARGET_ANDROID)

# build components
anakin_option(BUILD_WITH_UNIT_TEST "Build anakin unit test components." YES)

anakin_option(BUILD_WITH_LITE "Build anakin lite components." YES if NOT USE_ARM_PLACE)

# build examples
anakin_option(BUILD_EXAMPLES "build detection and classification examples" NO)

# build target
anakin_option(BUILD_SHARED "Build anakin shared lib." YES)
anakin_option(BUILD_STATIC "Build anakin static lib." YES if NOT BUILD_SHARED)

anakin_option(ENABLE_OP_TIMER "Enable op timer mode." NO)

# ----------------------------------------------------------------------------
# section: anakin compiler and linker options
# ----------------------------------------------------------------------------
if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug FORCE)
else()
    set(CMAKE_BUILD_TYPE Release FORCE)
endif()

if(USE_LOGGER) 
	anakin_option(ENABLE_STACKTRACES "If enable local logger with stacktrace." YES if NOT USE_ARM_PLACE)
	anakin_option(SUPPORT_PTHREADS "If enable local logger with supporting pthreads. " YES)
endif()

# ----------------------------------------------------------------------------
# section:configure a header file to pass some of the CMake settings to the source
# 		  code
# ----------------------------------------------------------------------------
configure_file (
        "${PROJECT_SOURCE_DIR}/cmake/config/anakin_config.h.in"
        "${PROJECT_BINARY_DIR}/anakin_config.h"
)
# add the binary tree to the search path so that anakin will find ak_config.h
include_directories(${PROJECT_BINARY_DIR})

# ----------------------------------------------------------------------------
# section: find modules and configure
# ----------------------------------------------------------------------------
set(ANAKIN_SABER_DEPENDENCIES) # set saber dependent third-party lib targets

# check and set base compile options
include(cmake/compiler_options.cmake)

include(cmake/find_modules.cmake)
if(USE_CUDA)
    include(cmake/cuda.cmake)
endif()

if(USE_X86_PLACE)
    set(ANAKIN_TEMP_THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/third-party)
    if(USE_MKLML)
        include(cmake/external/mklml.cmake)
    endif()
    if(USE_XBYAK)
        include(cmake/external/xbyak.cmake)
    endif()
    #include(cmake/external/mkldnn.cmake)
endif()

# gather all the config options to anakin
include(cmake/gather.cmake)


# ----------------------------------------------------------------------------
# section: build and install anakin 
# ----------------------------------------------------------------------------
# add source sub_directory whick holds the cmake build module
# fetch files of model_parser
add_subdirectory(${ANAKIN_MODEL_PARSER})
add_subdirectory(${ANAKIN_SABER})
add_subdirectory(${ANAKIN_FRAMEWORK})

if(BUILD_WITH_UNIT_TEST)
    add_subdirectory(${ANAKIN_UNIT_TEST})
endif()

if(BUILD_WITH_LITE)
	add_subdirectory(${ANAKIN_LITE})
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(${ANAKIN_EXAMPLES})
endif()

anakin_print_statistic()


#set(executable_output_path ${PROJECT_BINARY_DIR}/unit_test)

