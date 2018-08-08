# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.0)
project(ANAKIN-lite C CXX)

include(../../cmake/msg_color.cmake)
include(../../cmake/utils.cmake)

anakin_option(USE_ARM_PLACE "Select the build mode for ARM place." YES)

# options
anakin_option(USE_ANDROID "using android place." YES)
anakin_option(USE_IOS "using android place." NO if NOT USE_ANDROID)

anakin_option(USE_OMP "using openmp for lite." YES)
anakin_option(USE_LOGGER "using local logger" YES)

anakin_option(ENABLE_DEBUG "Enable DEBUG(default) mode." NO)

configure_file ( 
	"../../cmake/config/anakin_config.h.in" 
	"${PROJECT_BINARY_DIR}/anakin_config.h"
)

# add compile flags
anakin_add_compile_option(-std=c++11)
anakin_add_compile_option(-fPIC)
anakin_add_compile_option(-ldl)
anakin_add_compile_option(-fvisibility=hidden)
anakin_add_compile_option(-fvisibility-inlines-hidden)
anakin_add_compile_option(-ffunction-sections)
anakin_add_compile_option(-fdata-sections)
anakin_add_compile_option(-s)
#anakin_add_compile_option(-fstrict-aliasing)

#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now ${CMAKE_SHARED_LINKER_FLAGS}")
#set(CMAKE_MODULE_LINKER_FLAGS "-Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now ${CMAKE_MODULE_LINKER_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS    "-Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now ${CMAKE_EXE_LINKER_FLAGS}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe")
set(MAKE_STATIC_LINKER_FLAGS "${MAKE_STATIC_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe")

anakin_add_compile_option(-W)
anakin_add_compile_option(-Wall)
anakin_add_compile_option(-Wno-unused-variable) # no unused-variable
anakin_add_compile_option(-Wformat)
anakin_add_compile_option(-Wmissing-declarations)
anakin_add_compile_option(-Winit-self)
anakin_add_compile_option(-Wpointer-arith)
anakin_add_compile_option(-Wshadow)
anakin_add_compile_option(-fpermissive)
anakin_add_compile_option(-Wsign-promo)
anakin_add_compile_option(-fdiagnostics-show-option) 
anakin_add_compile_option(-Wno-undef) 
anakin_add_compile_option(-Wno-narrowing) 
anakin_add_compile_option(-Wno-unknown-pragmas) 
anakin_add_compile_option(-Wno-delete-non-virtual-dtor) 
anakin_add_compile_option(-Wno-comment) 
anakin_add_compile_option(-Wno-sign-compare) 
anakin_add_compile_option(-Wno-ignored-qualifiers) 
anakin_add_compile_option(-Wno-enum-compare)

if(ENABLE_DEBUG)
	set(CMAKE_BUILD_TYPE Debug FORCE)
else()
	set(CMAKE_BUILD_TYPE Release FORCE)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug) 
	anakin_add_compile_option(-O0) 
	anakin_add_compile_option(-g) 
else() 
	anakin_add_compile_option(-Os) 
	anakin_add_compile_option(-DNDEBUG) 
endif()

if(USE_ANDROID) 
	anakin_add_compile_option(-pie) 
	anakin_add_compile_option(-mfloat-abi=softfp) 
	anakin_add_compile_option(-mfpu=neon) 
	anakin_add_compile_option(-ffast-math) 
	anakin_add_compile_option(-lc) 
	set(ANAKIN_EXTRA_CXX_FLAGS "${ANAKIN_EXTRA_CXX_FLAGS} ${ANDROID_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS  ${ANAKIN_EXTRA_CXX_FLAGS})

message(STATUS "cmake flags: ${CMAKE_CXX_FLAGS}")

if(USE_IOS)
endif()

if(USE_OMP) 
	find_package(OpenMP REQUIRED) 
	if(OPENMP_FOUND OR OpenMP_CXX_FOUND) 
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}") 
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		message(STATUS "Found openmp in ${OPENMP_INCLUDE_DIR}") 
		message(STATUS " |-- openmp c flags:  ${OpenMP_C_FLAGS}") 
		message(STATUS " |-- openmp cxx flags:  ${OpenMP_CXX_FLAGS}") 
		message(STATUS " `-- openmp link flags:  ${OpenMP_EXE_LINKER_FLAGS}") 
		include_directories(${OPENMP_INCLUDE_DIR}) 
	else()
		message(FATAL_ERROR "Could not found openmp !") 
	endif() 
endif()

set(anakin_lite_lib_so "anakin_lite")
set(anakin_lite_lib_static "anakin_lite_static")

set(ANAKIN_LITE_ROOT ${PROJECT_SOURCE_DIR})

anakin_fetch_include_recursively(${ANAKIN_LITE_ROOT})

anakin_fetch_files_with_suffix(${ANAKIN_LITE_ROOT}/ "cpp" ANAKIN_LITE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_LITE_ROOT}/saber/lite/core "cpp" ANAKIN_SABER_ARM_LITE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_LITE_ROOT}/saber/lite/funcs "cpp" ANAKIN_SABER_ARM_LITE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_LITE_ROOT}/saber/lite/funcs/neon "cpp" ANAKIN_SABER_ARM_LITE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_LITE_ROOT}/saber/lite/funcs/neon/impl "cpp" ANAKIN_SABER_ARM_LITE_SRC)

add_library(ANAKIN_LITE_OBJS OBJECT ${ANAKIN_LITE_SRC} ${ANAKIN_SABER_ARM_LITE_SRC})

add_library(${anakin_lite_lib_so} SHARED $<TARGET_OBJECTS:ANAKIN_LITE_OBJS>)
set_target_properties(${anakin_lite_lib_so} PROPERTIES 
					  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/)

add_library(${anakin_lite_lib_static} STATIC $<TARGET_OBJECTS:ANAKIN_LITE_OBJS>)
set_target_properties(${anakin_lite_lib_static} PROPERTIES 
					  ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/)

install(DIRECTORY ${ANAKIN_LITE_ROOT}/saber  ${ANAKIN_LITE_ROOT}/utils
		DESTINATION ${PROJECT_SOURCE_DIR}/output/
		FILES_MATCHING
		PATTERN "*.bin"
		PATTERN "*.h"
		PATTERN "*.inl")
