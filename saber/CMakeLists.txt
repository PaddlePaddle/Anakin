# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# @file     CMakeLists files in the saber directory of project
# @auther   cuichaowen
# @date     2017-10-24
# ----------------------------------------------------------------------------

set(ANAKIN_SABER_STATIC_RELAY "" )

set(ANAKIN_SABER_BASE_SRC "")
anakin_fetch_include_recursively(${ANAKIN_SABER})

# add ak_base_source files
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core "cpp" ANAKIN_SABER_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/ "cpp" ANAKIN_SABER_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs "cpp" ANAKIN_SABER_BASE_SRC)

if(USE_ARM_PLACE)
    anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/arm "cpp" ANAKIN_SABER_BASE_SRC) 
	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/arm "cpp" ANAKIN_SABER_BASE_SRC) 
	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/arm/impl "cpp" ANAKIN_SABER_BASE_SRC)
endif()

if(USE_GPU_PLACE)
    if(USE_CUDA) 
        anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/cuda "cpp" ANAKIN_SABER_BASE_SRC)
        anakin_fetch_files_with_suffix(${ANAKIN_SABER}/funcs/impl/cuda "cpp" ANAKIN_SABER_BASE_SRC)
    else()
        message(ERROR "Use GPU place but not use cuda, not support amd yet.")
    endif()
endif()


if(USE_X86_PLACE)
	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/core/impl/x86 "cpp" ANAKIN_SABER_BASE_SRC)
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

if(USE_ARM_PLACE) 
	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/lite/core "cpp" ANAKIN_SABER_ARM_LITE_SRC)
	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/lite/funcs "cpp" ANAKIN_SABER_ARM_LITE_SRC)
	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/lite/funcs/neon "cpp" ANAKIN_SABER_ARM_LITE_SRC)
	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/lite/funcs/neon/impl "cpp" ANAKIN_SABER_ARM_LITE_SRC)
endif()

# add saber library to share
if(UNIX OR APPLE) 
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
endif()


set(ANAKIN_SABER_LIB_TARGET ${ANAKIN_SABER_TEMP_COMMMON_LIB} PARENT_SCOPE)
