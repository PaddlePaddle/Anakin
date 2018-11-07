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

################################################################################
macro(amd_set_opencl_path)
    if(NOT DEFINED OpenCL_INCLUDE_DIR)
        set(OpenCL_INCLUDE_DIR "/opt/rocm/opencl/include")
    endif()
    if(NOT DEFINED OpenCL_LIBRARY)
        set(OpenCL_LIBRARY "/opt/rocm/opencl/lib/x86_64/libOpenCL.so")
    endif()
    
    #FIND_PACKAGE(OpenCL REQUIRED)
    #if(OpenCL_FOUND)
    #    message(STATUS "Found OpenCL in ${OpenCL_INCLUDE_DIRS}")
    #    message(STATUS "Found OpenCL lib in ${OpenCL_LIBRARIES}")
    #    include_directories(${OpenCL_INCLUDE_DIRS})
    #    LINK_LIBRARIES(${OpenCL_LIBRARIES})
    #endif()
endmacro()

macro(amd_find_file regexps)
    FILE(GLOB FOUND_FILES ${regexps})
endmacro()

function(add_kernels KERNEL_FILES)
    set(INIT_KERNELS_LIST)

    foreach(KERNEL_FILE ${KERNEL_FILES})
        if("${CMAKE_VERSION}" VERSION_LESS 3.0)
            configure_file(${KERNEL_FILE} ${KERNEL_FILE}.delete)
        else()
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KERNEL_FILE})
        endif()
        get_filename_component(BASE_NAME ${KERNEL_FILE} NAME_WE)
        string(TOUPPER "${BASE_NAME}" KEY_NAME)
        string(MAKE_C_IDENTIFIER "${KEY_NAME}" VAR_NAME)
        list(APPEND INIT_KERNELS_LIST "    { \"${KEY_NAME}\", std::string(reinterpret_cast<const char*>(${VAR_NAME}), ${VAR_NAME}_SIZE) }")
    endforeach()
    string(REPLACE ";" ",\n" INIT_KERNELS "${INIT_KERNELS_LIST}")
    configure_file("${CMAKE_SOURCE_DIR}/saber/core/impl/amd/utils/amd_kernels.cpp.in" ${PROJECT_BINARY_DIR}/amd_kernels.cpp)
endfunction()

macro(generate_amd_kernel_src)
    set(AMD_KERNELS)
    set(AMD_KERNEL_INCLUDES)

    set(FILE_REGEXP
        "${CMAKE_SOURCE_DIR}/saber/funcs/impl/amd/cl/*.cl"
        "${CMAKE_SOURCE_DIR}/saber/funcs/impl/amd/cl/*.s"
        "${CMAKE_SOURCE_DIR}/saber/funcs/impl/amd/lib/*.so"
        "${CMAKE_SOURCE_DIR}/test/saber/amd/*.cl"
    )
    amd_find_file("${FILE_REGEXP}")
    list(APPEND AMD_KERNELS ${FOUND_FILES})
    
    set(FILE_REGEXP
        "${CMAKE_SOURCE_DIR}/saber/funcs/impl/amd/cl/*.inc"
    )
    amd_find_file("${FILE_REGEXP}")
    list(APPEND AMD_KERNEL_INCLUDES ${FOUND_FILES})
    
    #message(STATUS =======${AMD_KERNELS}======)
    #message(STATUS =======${AMD_KERNEL_INCLUDES}======)
    add_kernels("${AMD_KERNELS}")
    add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/amd_kernels.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${MIOPEN_PROJECT} ${AMD_KERNELS} ${AMD_KERNEL_INCLUDES}
        COMMAND ${MIOPEN_BINARY_DIR}/bin/addkernels -guard GUARD_AMD_KERNELS_HPP_ -target ${PROJECT_BINARY_DIR}/amd_kernels.h -source ${AMD_KERNELS}
        COMMENT "Inlining AMD kernels"
        )
    add_custom_target(amd_kernels DEPENDS ${PROJECT_BINARY_DIR}/amd_kernels.h)
    include_directories($PROJECT_BINARY_DIR}/amd_kernels.h)
    list(APPEND ANAKIN_SABER_DEPENDENCIES amd_kernels)
    list(APPEND ANAKIN_SABER_BASE_SRC ${PROJECT_BINARY_DIR}/amd_kernels.cpp)
endmacro()

macro(amd_set_miopengemm_path)
    if(NOT DEFINED MIOPENGEMM_INCLUDE_DIR)
        set(MIOPENGEMM_INCLUDE_DIR "/opt/rocm/miopengemm/include")
    endif()
    if(NOT DEFINED MIOPENGEMM_LIBRARY)
        set(MIOPENGEMM_LIBRARY "/opt/rocm/miopengemm/lib64/libmiopengemm.so")
    endif()

    #FIND_PACKAGE(miopengemm REQUIRED)
    #if(miopengemm_FOUND)
    #    message(STATUS "Found miopengemm in ${MIOPENGEMM_INCLUDE_DIR}")
    #    message(STATUS "Found miopengemm lib in ${MIOPENGEMM_LIBRARY}")
    #    include_directories(${MIOPENGEMM_INCLUDE_DIR})
    #    LINK_LIBRARIES(${MIOPENGEMM_LIBRARY})
    #endif()
endmacro()
