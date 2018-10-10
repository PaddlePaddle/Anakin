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

set(ANAKIN_SABER_STATIC_RELAY "" )

set(ANAKIN_SABER_BASE_SRC "")
anakin_fetch_include_recursively(${ANAKIN_SABER})
anakin_fetch_include_recursively(${ANAKIN_UTILS})

# add ak_base_source files
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core "cpp" ANAKIN_SABER_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/ "cpp" ANAKIN_SABER_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs "cpp" ANAKIN_SABER_BASE_SRC)

if(USE_ARM_PLACE)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/arm "cpp" ANAKIN_SABER_BASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/arm "cpp" ANAKIN_SABER_BASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/arm/impl "cpp" ANAKIN_SABER_BASE_SRC)
endif()

if(USE_BM_PLACE)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/bm "cpp" ANAKIN_SABER_BASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/bm "cpp" ANAKIN_SABER_BASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/bm/impl "cpp" ANAKIN_SABER_BASE_SRC)
endif()

if(USE_GPU_PLACE)
    if(USE_CUDA) 
        anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/cuda "cpp" ANAKIN_SABER_BASE_SRC)
        anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/cuda "cpp" ANAKIN_SABER_BASE_SRC)
    else()
        anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/amd "cpp" ANAKIN_SABER_BASE_SRC)
        anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/amd "cpp" ANAKIN_SABER_BASE_SRC)
    endif()
endif()


if(USE_X86_PLACE OR USE_BM_PLACE)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/x86 "cpp" ANAKIN_SABER_BASE_SRC)
endif()

if(USE_X86_PLACE)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/x86 "cpp" ANAKIN_SABER_BASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/x86/kernel "cpp" ANAKIN_SABER_BASE_SRC)
endif()

# compile cpp objs
# add_library(ANAKIN_SABER_BASE_OBJS OBJECT ${ANAKIN_SABER_BASE_SRC})

# whole archive for static lib
if(NOT MSVC AND NOT APPLE)
    set(WHOLE_ARCHIVE_START -Wl,--whole-archive)
    set(WHOLE_ARCHIVE_END -Wl,--no-whole-archive)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # using regular Clang or AppleClang
    set(WHOLE_ARCHIVE_START -Wl,-force_load)
    set(WHOLE_ARCHIVE_END)
endif()

set(ANAKIN_SABER_TEMP_COMMMON_LIB "anakin_saber_common")

if(USE_CUDA)
	# set root
    set(CUDA_BASE_CODE_ROOT ${ANAKIN_SABER}/funcs/impl/cuda/base)
	# set select arch for cuda
	add_subdirectory(${ANAKIN_SABER}/funcs/impl/cuda/base)

	set(FLAGS_BACKUP ${CMAKE_CXX_FLAGS}) 
	set(CMAKE_CXX_FLAGS "")
	if(BUILD_SHARED)
    		CUDA_COMPILE(ANAKIN_SABER_CUDA_C_SRC_OBJS SHARED ${ANAKIN_SABER_CUDA_C_SRC} OPTIONS ${ANAKIN_NVCC_FLAG})
	endif()
	if(BUILD_STATIC)
		CUDA_COMPILE(ANAKIN_SABER_CUDA_C_SRC_OBJS STATIC ${ANAKIN_SABER_CUDA_C_SRC} OPTIONS ${ANAKIN_NVCC_FLAG})
	endif()
    	set(CMAKE_CXX_FLAGS ${FLAGS_BACKUP})

	set(ANAKIN_SABER_STATIC_RELAY ${ANAKIN_SABER_STATIC_RELAY} 
				      ${BEGIN_WHOLE_ARCHIVE} 
				      ${ANAKIN_SABER_SASS_STATIC_LIB} 
				      ${WHOLE_ARCHIVE_END})	
endif()

if(USE_BM_PLACE)
    set(BIN_NAME bmkernel_bin)
    #set(LINK_CONFIG link/bm1682_ddr.lds)
    add_custom_command(OUTPUT bm_kernel_tmp
        COMMAND gcc -std=c99 -I${BM_ROOT}/include/config -I${BM_ROOT}/include/common -I${BM_ROOT}/include/c_model -I${BM_ROOT}/include/firmware_core -I${BM_ROOT}/include/bmlib -g -MMD -c -o ${BIN_NAME}.o ${ANAKIN_SABER}/funcs/impl/bm/device/bmkernel_base.c
        COMMAND AR ${BIN_NAME}.a
        COMMAND ar rcs ${BIN_NAME}.a ${BIN_NAME}.o
        COMMENT "BM Kernel compilation..."
    )
    add_custom_target(ANAKIN ALL DEPENDS bm_kernel_tmp)
    
    set(BM_KERNEL_DEVICE_LIBRARIES "")
	list(APPEND BM_KERNEL_DEVICE_LIBRARIES ${BIN_NAME}.a)
	list(APPEND ANAKIN_LINKER_LIBS ${BM_KERNEL_DEVICE_LIBRARIES})
endif()

# add saber library to static
if(UNIX OR APPLE)
    if (USE_ARM_PLACE)
        ADD_LIBRARY(${ANAKIN_SABER_TEMP_COMMMON_LIB} STATIC ${ANAKIN_SABER_CUDA_C_SRC_OBJS} ${ANAKIN_SABER_BASE_SRC})
        set_target_properties(${ANAKIN_SABER_TEMP_COMMMON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                ${ANAKIN_ROOT}/output/)
    else()
        if (BUILD_SHARED)
            ADD_LIBRARY(${ANAKIN_SABER_TEMP_COMMMON_LIB} SHARED ${ANAKIN_SABER_CUDA_C_SRC_OBJS} ${ANAKIN_SABER_BASE_SRC})
            #$<TARGET_OBJECTS:ANAKIN_SABER_BASE_OBJS>)
            if(USE_X86_PLACE)
                message(STATUS ${ANAKIN_SABER_DEPENDENCIES})
                add_dependencies(${ANAKIN_SABER_TEMP_COMMMON_LIB} ${ANAKIN_SABER_DEPENDENCIES})
            endif()
            set_target_properties(${ANAKIN_SABER_TEMP_COMMMON_LIB} PROPERTIES VERSION ${VERSION})
            target_link_libraries(${ANAKIN_SABER_TEMP_COMMMON_LIB} ${ANAKIN_LINKER_LIBS})
            target_link_libraries(${ANAKIN_SABER_TEMP_COMMMON_LIB} ${ANAKIN_SABER_STATIC_RELAY})
            set_target_properties(${ANAKIN_SABER_TEMP_COMMMON_LIB} PROPERTIES LINK_FLAGS "")
            set_target_properties(${ANAKIN_SABER_TEMP_COMMMON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                    ${ANAKIN_ROOT}/output/)
        else()
            ADD_LIBRARY(${ANAKIN_SABER_TEMP_COMMMON_LIB} STATIC ${ANAKIN_SABER_CUDA_C_SRC_OBJS} ${ANAKIN_SABER_BASE_SRC})
            set_target_properties(${ANAKIN_SABER_TEMP_COMMMON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                    ${ANAKIN_ROOT}/output/)
        endif ()
    endif()
endif()
set(ANAKIN_SABER_LIB_TARGET ${ANAKIN_SABER_TEMP_COMMMON_LIB} PARENT_SCOPE)
