# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# @file     CMakeLists files in the saber directory of project
# @auther   cuichaowen
# @date     2017-10-24
# ----------------------------------------------------------------------------

set(ANAKIN_SABER_LIB "" )

set(ANAKIN_SABER_BASE_SRC "")
anakin_fetch_include_recursively(${ANAKIN_SABER})

# add ak_base_source files
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core "cpp" ANAKIN_SABER_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/ "cpp" ANAKIN_SABER_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs "cpp" ANAKIN_SABER_BASE_SRC)

if(USE_ARM_PLACE)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/arm "cpp" ANAKIN_SABER_BASE_SRC)
endif()

if(USE_GPU_PLACE)
    if(USE_CUDA) 
        anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/cuda "cpp" ANAKIN_SABER_BASE_SRC)
        anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/cuda "cpp" ANAKIN_SABER_BASE_SRC)
    else()
        message(ERROR "Use GPU place but not use cuda, not support amd yet.")
    endif()
endif()


    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/x86 "cpp" ANAKIN_SABER_BASE_SRC)
if(USE_X86_PLACE)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/x86 "cpp" ANAKIN_SABER_BASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/x86/kernel "cpp" ANAKIN_SABER_BASE_SRC)
endif()

# compile cpp objs
add_library(ANAKIN_SABER_BASE_OBJS OBJECT ${ANAKIN_SABER_BASE_SRC})

if(USE_CUDA)
	# set root
    set(CUDA_BASE_CODE_ROOT ${ANAKIN_SABER}/funcs/impl/cuda/base)
	# set select arch for cuda
	add_subdirectory(${ANAKIN_SABER}/funcs/impl/cuda/base)

	set(ANAKIN_SABER_TEMP_COMMMON_LIB "anakin_saber_common_static")
	set(FLAGS_BACKUP ${CMAKE_CXX_FLAGS}) 
	set(CMAKE_CXX_FLAGS "")
	if(BUILD_SHARED)
    	CUDA_COMPILE(ANAKIN_SABER_CUDA_C_SRC_OBJS SHARED ${ANAKIN_SABER_CUDA_C_SRC} OPTIONS ${ANAKIN_NVCC_FLAG})
	endif()
	if(BUILD_STATIC)
		CUDA_COMPILE(ANAKIN_SABER_CUDA_C_SRC_OBJS STATIC ${ANAKIN_SABER_CUDA_C_SRC} OPTIONS ${ANAKIN_NVCC_FLAG})
	endif()
    set(CMAKE_CXX_FLAGS ${FLAGS_BACKUP})
	# add saber library to static
	if(UNIX OR APPLE)
		ADD_LIBRARY(${ANAKIN_SABER_TEMP_COMMMON_LIB} STATIC ${ANAKIN_SABER_CUDA_C_SRC_OBJS} 
									 $<TARGET_OBJECTS:ANAKIN_SABER_BASE_OBJS>) 
		set_target_properties(${ANAKIN_SABER_TEMP_COMMMON_LIB} PROPERTIES VERSION ${VERSION})
		target_link_libraries(${ANAKIN_SABER_TEMP_COMMMON_LIB} ${ANAKIN_LINKER_LIBS})
		set_target_properties(${ANAKIN_SABER_TEMP_COMMMON_LIB} PROPERTIES LINK_FLAGS "")
		set_target_properties(${ANAKIN_SABER_TEMP_COMMMON_LIB} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY 
						  		       ${PROJECT_BINARY_DIR}/)
	endif()
endif()

set(ANAKIN_SABER_LIB_TARGET ${ANAKIN_SABER_TEMP_COMMMON_LIB} PARENT_SCOPE)
set(ANAKIN_SABER_LIB "-Wl,-whole-archive ${PROJECT_BINARY_DIR}/lib${ANAKIN_SABER_TEMP_COMMMON_LIB}.a -Wl,--no-whole-archive") 
if(USE_CUDA)
    set(ANAKIN_SABER_LIB "${ANAKIN_SABER_LIB} -Wl,-whole-archive ${ANAKIN_SABER_SASS_STATIC_LIB} -Wl,--no-whole-archive" PARENT_SCOPE)
else()
    set(ANAKIN_SABER_LIB ${ANAKIN_SABER_LIB} PARENT_SCOPE)
endif()
