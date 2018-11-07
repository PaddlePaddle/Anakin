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
# section: Set nvcc arch info.
# ----------------------------------------------------------------------------
function(anakin_set_nvcc_archs_info arch_list)
	set(__nvcc_arch_flag "")
	foreach(arch ${${arch_list}})	
		message(STATUS " `--support arch :  ${BoldWhite}${arch}${ColourReset}")
		string(REPLACE "." ";" arch ${arch})
		list(GET arch 0 major)
		list(GET arch 1 minor)
		set(__ARCH "${major}${minor}")
		if(NOT __nvcc_arch_flag MATCHES "_${__ARCH}")
			set(__nvcc_arch_flag "${__nvcc_arch_flag} --generate-code arch=compute_${__ARCH},code=sm_${__ARCH} ")	
		endif()
		set(ANAKIN_NVCC_FLAG " ${ANAKIN_NVCC_FLAG} ${__nvcc_arch_flag}" PARENT_SCOPE)
	endforeach()
endfunction()	

# ----------------------------------------------------------------------------
# section: Detect the cuda GPU SM capabiities.
# ----------------------------------------------------------------------------
macro(anakin_detect_arch)
	set(__filename ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CUDA_TEMP/detect_sm_cap.cu)
	set(__working_dir ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CUDA_TEMP/)
	file(WRITE ${__filename} ""
     	"#include <cstdio>\n"
     	"int main() {\n"
     	"  int count = 0;\n"
     	"  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
     	"  if (count == 0) return -1;\n"
     	    "  printf(\"%d\",count);\n"
     	"  for (int device = 0; device < count; ++device) {\n"
     	"    cudaDeviceProp prop;\n"
     	"    if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {\n"
     	"      printf(\";%d.%d\", prop.major, prop.minor);\n"
     	"    }\n"
     	"  }\n"
     	"  return 0;\n"
     	"}\n")
	execute_process(COMMAND nvcc --run -Wno-deprecated-gpu-targets ${__filename}
			WORKING_DIRECTORY ${__working_dir} 
			OUTPUT_VARIABLE __arch_detection_out
			RESULT_VARIABLE _ret_arch_detection_out)
	if(NOT _ret_arch_detection_out)
		list(GET __arch_detection_out 0 CUDA_DEV_COUNT)	
		message(STATUS "Detected ${BoldWhite}${CUDA_DEV_COUNT}${ColourReset} cuda device !")
		list(REMOVE_AT __arch_detection_out  0)
		foreach(arch ${__arch_detection_out})
			message(STATUS "         `-- ${BoldWhite}Arch ${arch}${ColourReset}")
			string(REPLACE "." ";" arch_l ${arch})	
			list(GET arch_l 0 major)
			list(GET arch_l 1 minor)
			set(__ARCH "${major}${minor}")
			if(NOT (${ANAKIN_NVCC_FLAG} MATCHES "_${__ARCH}"))
                set(ANAKIN_NVCC_FLAG  "${ANAKIN_NVCC_FLAG} --generate-code arch=compute_${__ARCH},code=sm_${__ARCH} ")
            endif()
		endforeach()
	else()
		message(WARNING "Couldn't detect the local GPU machine .")
	endif()
endmacro()

# ----------------------------------------------------------------------------
# section: Find cudnn.
# ----------------------------------------------------------------------------
macro(anakin_find_cudnn)
	set(CUDNN_ROOT "" CACHE PATH "CUDNN root dir.")
  	find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS ${CUDNN_ROOT} 
						  $ENV{CUDNN_ROOT} 
						  $ENV{CUDNN_ROOT}/include
						  ${ANAKIN_ROOT}/third-party/cudnn/include NO_DEFAULT_PATH)
    if(BUILD_SHARED)
        find_library(CUDNN_LIBRARY NAMES libcudnn.so 
                               PATHS ${CUDNN_INCLUDE_DIR}/../lib64/ ${CUDNN_INCLUDE_DIR}/
                               DOC "library path for cudnn.") 
    else()
        find_library(CUDNN_LIBRARY NAMES libcudnn_static.a
                               PATHS ${CUDNN_INCLUDE_DIR}/../lib64/
                               DOC "library path for cudnn.")
    endif()
 
	if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
		set(CUDNN_FOUND YES)
		file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_FILE_VERSION)
		string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
       			CUDNN_VERSION_MAJOR "${CUDNN_FILE_VERSION}")
    		string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
       			CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    		string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
       			CUDNN_VERSION_MINOR "${CUDNN_FILE_VERSION}")
    		string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
       			CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    		string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
       			CUDNN_VERSION_PATCH "${CUDNN_FILE_VERSION}")
    		string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
       			CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
		#message(STATUS "Found cudnn version ${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
		set(Cudnn_VERSION ${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH})
		string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 6 CUDNN_TOO_LOW)
		if(CUDNN_TOO_LOW)
			message(FATAL_ERROR " Cudnn version should > 6 ")
		endif()
	endif()
	if(CUDNN_FOUND)
		include_directories(SYSTEM ${CUDNN_INCLUDE_DIR})	
		list(APPEND ANAKIN_LINKER_LIBS ${CUDNN_LIBRARY})
		message(STATUS "Found cudnn: ${CUDNN_INCLUDE_DIR}")
	else()
		message(SEND_ERROR "Could not find cudnn library in: ${CUDNN_ROOT}")
	endif()
endmacro()

# ----------------------------------------------------------------------------
# section: Find cuda and config compile options.
# ----------------------------------------------------------------------------
macro(anakin_find_cuda)
	if(ENABLE_VERBOSE_MSG)
    	set(CUDA_VERBOSE_BUILD ON)
    endif()
    if(BUILD_CUBIN)
    	set(CUDA_BUILD_CUBIN ON) # defauld OFF
    endif()
	find_package(CUDA 7.5 REQUIRED)
    set(CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
    if(BUILD_SHARED)
	    if(CUDA_FOUND)
	    	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
	    	if(USE_CUBLAS)
	    		list(APPEND ANAKIN_LINKER_LIBS ${CUDA_CUBLAS_LIBRARIES}) 
	    	endif()
	    	if(USE_CURAND)
	    		list(APPEND ANAKIN_LINKER_LIBS ${CUDA_curand_LIBRARY})
	    	endif()
            if(BUILD_RPC) 
                list(APPEND ANAKIN_LINKER_LIBS ${CUDA_INCLUDE_DIRS}/../lib64/stubs/libnvidia-ml.so) 
            endif()
	    	list(APPEND ANAKIN_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
	    else()
	    	message(FATAL_ERROR "Cuda SHARED lib Could not found !")	
	    endif()
    else() # BUILD_STATIC
        find_path(CUDA_INCLUDE_DIRS cuda.h PATHS /usr/local/cuda/include 
												 /usr/include) 
        if(CUDA_INCLUDE_DIRS)
            include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
            find_library(CUDA_LIBRARY NAMES libcudart_static.a
                                   PATHS ${CUDA_INCLUDE_DIRS}/../lib64/
                                   DOC "library path for cuda.")
            if(CUDA_LIBRARY)
                list(APPEND ANAKIN_LINKER_LIBS ${CUDA_INCLUDE_DIRS}/../lib64/libcudart_static.a)
                if(USE_CUBLAS)
                    list(APPEND ANAKIN_LINKER_LIBS ${CUDA_INCLUDE_DIRS}/../lib64/libcublas_static.a)
                    list(APPEND ANAKIN_LINKER_LIBS ${CUDA_INCLUDE_DIRS}/../lib64/libcublas_device.a)
                endif()
                if(USE_CURAND)
                    list(APPEND ANAKIN_LINKER_LIBS ${CUDA_INCLUDE_DIRS}/../lib64/libcurand_static.a)
                endif()
                list(APPEND ANAKIN_LINKER_LIBS ${CUDA_INCLUDE_DIRS}/../lib64/libculibos.a)
            endif()
        else()
            message(ERROR "Cuda STATIC lib Could not Found !")
        endif()
    endif()

	# build cuda part for local machine.
    if(BUILD_CROSS_PLANTFORM)
        if(BUILD_FAT_BIN)
		    message(STATUS "Building fat-bin for cuda code !")
		    anakin_set_nvcc_archs_info(ANAKIN_ARCH_LIST)
        else()
            message(STATUS "Building cross-plantform target for cuda code !")
			anakin_detect_arch()
        endif()
	else()
		anakin_set_nvcc_archs_info(TARGET_GPUARCH)
    endif()
endmacro()
