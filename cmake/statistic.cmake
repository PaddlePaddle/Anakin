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

# ----------------------------------------------------------------------------
# section: prints the statistic of configuration of anakin.
# ----------------------------------------------------------------------------

function(anakin_print_statistic)
	message(STATUS "")
  	message(STATUS "================================ configuration ==================================")
  	message(STATUS "${Green}General:${ColourReset}")
  	message(STATUS "  anakin version            : ${BoldWhite}${VERSION}${ColourReset}")
  	message(STATUS "  System                    : ${BoldWhite}${CMAKE_SYSTEM_NAME}${ColourReset}")
  	message(STATUS "  C++ compiler              : ${BoldWhite}${CMAKE_CXX_COMPILER}${ColourReset}")
	message(STATUS "  C flags                   : ${CMAKE_C_FLAGS}")
  	message(STATUS "  CXX flags                 : ${CMAKE_CXX_FLAGS}")
	message(STATUS "  Link flags                : ${CMAKE_EXE_LINKER_FLAGS}")
	message(STATUS "  Shared Link flags         : ${CMAKE_SHARED_LINKER_FLAGS}")
	message(STATUS "  Anakin Link Libs          : ${ANAKIN_LINKER_LIBS}")
  	message(STATUS "  Build type                : ${BoldWhite}${CMAKE_BUILD_TYPE}${ColourReset}")
	message(STATUS "  Build cross plantform     : ${BUILD_CROSS_PLANTFORM}")
	if(ANAKIN_TYPE_FP64)
	message(STATUS "  Build anakin fp64         : ${ANAKIN_TYPE_FP64}")
	elseif(ANAKIN_TYPE_FP32)
	message(STATUS "  Build anakin fp32         : ${ANAKIN_TYPE_FP32}")
	elseif(ANAKIN_TYPE_FP16)
	message(STATUS "  Build anakin fp16         : ${ANAKIN_TYPE_FP16}")
	elseif(ANAKIN_TYPE_INT8)
	message(STATUS "  Build anakin int8         : ${ANAKIN_TYPE_INT8}")
	else()
	message(STATUS "  Build anakin type         : ${BoldRed}Unknow${ColourReset}")
	endif()
  	message(STATUS "")
    if(BUILD_SHARED)
  	message(STATUS "  Build shared libs         : ${BUILD_SHARED}")
    else()
	message(STATUS "  Build static libs         : ${BUILD_STATIC}")
    endif()
	message(STATUS "  Build with unit test      : ${BUILD_WITH_UNIT_TEST}")
	message(STATUS "")
	message(STATUS "  Enable verbose message    : ${ENABLE_VERBOSE_MSG}")
	message(STATUS "  Enable noisy warnings     : ${ENABLE_NOISY_WARNINGS}")
	message(STATUS "  Disable all warnings      : ${DISABLE_ALL_WARNINGS}")

	message(STATUS "")
	if(USE_GLOG)
	message(STATUS "  USE_GLOG                  : ${USE_GLOG}")
	else()
	message(STATUS "  Use local logger          : logger")
	endif()
	if(USE_PROTOBUF)
	message(STATUS "  Use google protobuf       : ${USE_PROTOBUF}")
	endif()
	if(USE_NANOPB)
        message(STATUS "  USE nanopb                : ${USE_NANOPB}")
        endif()
	if(USE_GTEST)
	message(STATUS "  USE_GTEST                 : ${USE_GTEST}")
    else()
    message(STATUS "  Use local Unit test       : aktest")
    endif()

	message(STATUS "  USE_OPENCV                : ${USE_OPENCV}")
    if(USE_OPENCV)
    message(STATUS "    `-- OpenCV version      : ${OpenCV_VERSION}")
    endif()

	message(STATUS "  USE_BOOST                 : ${USE_BOOST}")
    if(USE_BOOST)
    message(STATUS "    `--Boost version        : ${BOOST_VERSION}")
    endif()	

    if(USE_MKL)
	message(STATUS "  USE Intel(R) MKL          : ${USE_MKL}")
    endif()	
    message(STATUS "  USE_OPENMP                : ${USE_OPENMP}")
    if(USE_OPENMP)
    message(STATUS "    `--Openmp version       : ${OPENMP_VERSION}")
    endif()	

	if(USE_X86_PLACE)
	message(STATUS "")
	message(STATUS "${Green}X86:${ColourReset}")
	message(STATUS "  USE_X86                  : ${USE_X86_PLACE}")
	message(STATUS "  X86 Target Arch          : ${BUILD_X86_ARCH}")
	endif()
	
    if(USE_CUDA)
	message(STATUS "")
	message(STATUS "${Green}Cuda:${ColourReset}")
  	message(STATUS "  USE_CUDA                  : ${USE_CUDA}")
  	if(USE_CUDA)
  	message(STATUS "    |--CUDA version         : ${CUDA_VERSION}")
	message(STATUS "    `--NVCC flags           : ${ANAKIN_NVCC_FLAG}")
  	endif()

	message(STATUS "  USE_CUBLAS                : ${USE_CUBLAS}")
    message(STATUS "  USE_CURAND                : ${USE_CURAND}")
	message(STATUS "  USE_CUFFT                 : ${USE_CUFFT}")
    message(STATUS "  USE_CUDNN                 : ${USE_CUDNN}")
	if(USE_CUDNN)
    message(STATUS "    `--Cudnn version        : ${Cudnn_VERSION}")
    endif()
	
	message(STATUS "")
  	message(STATUS "  USE_OPENCL                : ${USE_OPENCL}")
  	if(USE_OPENCL)
  	message(STATUS "    `--OpenCL version       : ${OpenCL_VERSION}")
  	endif()
    endif()
    

	message(STATUS "")
	if(USE_GPU_PLACE)
  	message(STATUS "  SELECT_GPU_PLACE          : ${USE_GPU_PLACE}")
	elseif(USE_X86_PLACE)
	message(STATUS "  SELECT_X86_PLACE          : ${USE_X86_PLACE}")
	elseif(USE_ARM_PLACE)
  	message(STATUS "  USE_ARM_PLACE             : ${USE_ARM_PLACE}")
    if(TARGET_ANDROID)
    message(STATUS "    `--Target Android       : ${TARGET_ANDROID}")
    else()
    message(STATUS "    `--Target IOS           : ${TARGET_IOS}")
    endif()
	else()
	message(STATUS "  Error select place!    ")
	endif()

	message(STATUS "")
  	message(STATUS "  Configuation path         : ${PROJECT_BINARY_DIR}/anakin_config.h")
	message(STATUS "================================ End ==================================")
endfunction()
