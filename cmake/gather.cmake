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

# find cudnn default cudnn 5
if(USE_CUDNN)
    anakin_find_cudnn()
endif()

# find cuda
if(USE_CUDA) 
    #set other cuda path
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
    anakin_find_cuda()
endif()

if(USE_BM)
    #anakin_find_bm()
endif()

# set amd opencl path
if(USE_AMD)
    #amd_set_opencl_path()
    #amd_build_cl_file("${CMAKE_SOURCE_DIR}/saber/core/impl/amd/" "${CMAKE_BINARY_DIR}/cl/amd/")
    amd_build_cl_file("${CMAKE_SOURCE_DIR}/saber/funcs/impl/amd/cl" "${CMAKE_BINARY_DIR}/cl/amd")
    amd_build_cl_binary_file("${CMAKE_SOURCE_DIR}/saber/funcs/impl/amd/lib" "${CMAKE_BINARY_DIR}/cl/amd")
    amd_build_cl_file("${CMAKE_SOURCE_DIR}/saber/funcs/impl/amd/cl" "${PROJECT_SOURCE_DIR}/output/unit_test")
    amd_build_cl_binary_file("${CMAKE_SOURCE_DIR}/saber/funcs/impl/amd/lib" "${PROJECT_SOURCE_DIR}/output/unit_test")
    amd_build_cl_file("${CMAKE_SOURCE_DIR}/test/saber/amd" "${PROJECT_SOURCE_DIR}/output/unit_test")
endif()

# find opencl
if(USE_OPENCL)
    #anakin_generate_kernel(${ANAKIN_ROOT})
    anakin_find_opencl()
endif()

# find opencv
if(USE_OPENCV)
    anakin_find_opencv()
endif()

# find boost default boost 1.59.0
if(USE_BOOST)
    anakin_find_boost()
endif()

if(BUILD_WITH_GLOG)
    anakin_find_glog()
endif()

if(USE_PROTOBUF)
    anakin_find_protobuf()
    anakin_protos_processing()
endif()

if (USE_GFLAGS)
    anakin_find_gflags()
endif()

if(USE_MKL)
    #anakin_find_mkl()
endif()

if (USE_XBYAK)
    #anakin_find_xbyak()
endif()
if (USE_MKLML)
    #anakin_find_mklml()
endif()

if(ENABLE_VERBOSE_MSG)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

if(DISABLE_ALL_WARNINGS) 
    anakin_disable_warnings(CMAKE_CXX_FLAGS)
endif()
if(USE_OPENMP)
    anakin_find_openmp()
endif()
if(USE_ARM_PLACE)
    if(TARGET_ANDROID)
		if(USE_OPENMP)
        	anakin_find_openmp()
		endif()
    elseif(TARGET_IOS)
        message(STATUS " TARGET_IOS error")
    else()
        message(FATAL_ERROR " ARM TARGET unknown !")
    endif()
endif()
