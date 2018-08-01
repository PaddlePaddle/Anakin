# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# ----------------------------------------------------------------------------

# options
option(USE_ARM_PLACE "Select the build mode for ARM place." YES)
option(USE_ARMV8 "build armv8" NO)
option(USE_ANDROID "using android place." YES)
option(USE_IOS "using android place." NO)
option(TARGET_IOS "using ios" NO)
option(USE_OPENMP "using openmp for lite." YES)
option(ENABLE_DEBUG "Enable DEBUG(default) mode." NO)
option(BUILD_LITE_UNIT_TEST "build unit test for lite." YES)

if (USE_ARMV8)
	set(ANDROID_ABI "arm64-v8a")
else()
	set(ANDROID_ABI "armeabi-v7a with NEON")
endif()

if(CMAKE_TOOLCHAIN_FILE)
	set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
	# get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
	get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
	find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
	message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
	set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 2.8)
project(ANAKIN-lite C CXX)

configure_file ( 
	"../../cmake/config/anakin_config.h.in" 
	"${PROJECT_BINARY_DIR}/anakin_config.h"
)

if(ENABLE_DEBUG)
	set(CMAKE_BUILD_TYPE Debug FORCE)
else()
	set(CMAKE_BUILD_TYPE Release FORCE)
	add_definitions(-Ofast)
	add_definitions(-ffast-math)
endif()

set(anakin_lite_lib_so "anakin_lite")
set(anakin_lite_lib_static "anakin_lite_static")

# disable shared library on xcode ios
if(USE_IOS)
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
endif()
add_definitions(-std=c++11)
add_definitions(-Wall -Wextra -Wno-unused-function)
add_definitions(-fPIC)
add_definitions(-fvisibility=hidden -fvisibility-inlines-hidden)

add_definitions(-fPIC)
add_definitions(-fvisibility=hidden)
add_definitions(-fvisibility-inlines-hidden)
if (USE_ANDROID)
add_definitions(-ldl)
add_definitions(-ffunction-sections)
add_definitions(-fdata-sections)
add_definitions(-s)
endif()
#anakin_add_compile_option(-fstrict-aliasing)
add_definitions(-W)
add_definitions(-Wall)
add_definitions(-Wno-unused-variable) # no unused-variable
add_definitions(-Wformat)
add_definitions(-Wmissing-declarations)
add_definitions(-Winit-self)
add_definitions(-Wpointer-arith)
add_definitions(-Wno-shadow)
add_definitions(-fpermissive)
add_definitions(-Wsign-promo)
add_definitions(-fdiagnostics-show-option)
add_definitions(-Wno-undef)
add_definitions(-Wno-narrowing)
add_definitions(-Wno-unknown-pragmas)
add_definitions(-Wno-delete-non-virtual-dtor)
add_definitions(-Wno-comment)
add_definitions(-Wno-sign-compare)
add_definitions(-Wno-ignored-qualifiers)
add_definitions(-Wno-enum-compare)
add_definitions(-Wno-unused-parameter)
add_definitions(-Wno-unused-function)

if(USE_ANDROID)
	add_definitions(-pie)
	if(USE_ARMV8)
	else()
		add_definitions(-mfloat-abi=softfp)
		add_definitions(-mfpu=neon)
	endif()
	add_definitions(-ffast-math)
	add_definitions(-lc)
endif()

if(USE_OPENMP)
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


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe")
set(MAKE_STATIC_LINKER_FLAGS "${MAKE_STATIC_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe")

set(ANAKIN_LITE_SABER ${CMAKE_CURRENT_SOURCE_DIR}/../../saber/lite)
set(UNIT_TEST_LITE ${CMAKE_CURRENT_SOURCE_DIR}/../../test/lite)

include_directories(${ANAKIN_LITE_SABER}/../)
include_directories(${ANAKIN_LITE_SABER}/../../)
include_directories(${PROJECT_BINARY_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

FILE(GLOB BUILD_SRC_FILES1 "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
FILE(GLOB BUILD_SRC_FILES2 "${ANAKIN_LITE_SABER}/core/*.cpp")
FILE(GLOB BUILD_SRC_FILES3 "${ANAKIN_LITE_SABER}/funcs/*.cpp")
FILE(GLOB BUILD_SRC_FILES4 "${ANAKIN_LITE_SABER}/funcs/neon/*.cpp")
FILE(GLOB BUILD_SRC_FILES5 "${ANAKIN_LITE_SABER}/funcs/neon/impl/*.cpp")
if(USE_ANDROID)
	FILE(GLOB UNIT_TEST_LITE_SRC "${UNIT_TEST_LITE}/*.cpp")
elseif(USE_IOS)
	FILE(GLOB UNIT_TEST_LITE_SRC "${UNIT_TEST_LITE}/test_lite_model_ios.cpp")
endif()

add_library(ANAKIN_LITE_OBJS OBJECT ${BUILD_SRC_FILES1} ${BUILD_SRC_FILES2} ${BUILD_SRC_FILES3} ${BUILD_SRC_FILES4} ${BUILD_SRC_FILES5})

if(USE_ANDROID)
	add_library(${anakin_lite_lib_so} SHARED $<TARGET_OBJECTS:ANAKIN_LITE_OBJS>)
	set_target_properties(${anakin_lite_lib_so} PROPERTIES
					  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/)
endif()
add_library(${anakin_lite_lib_static} STATIC $<TARGET_OBJECTS:ANAKIN_LITE_OBJS>)
set_target_properties(${anakin_lite_lib_static} PROPERTIES 
					  ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/)

if(BUILD_LITE_UNIT_TEST)
	add_definitions(-fexceptions)
	# build test cases
	foreach(SRC_NAME ${UNIT_TEST_LITE_SRC})
		#unpack the dir "/"
		string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
		list(GET SEXY_LIST -1 TEST_CASE_NAME)
		#get the file name without suffix
		string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
		list(GET SEXY_LIST 0 TEST_CASE_NAME)
		add_executable(${TEST_CASE_NAME}  ${SRC_NAME})
		if(NO)#BUILD_SHARED)
			target_link_libraries(${TEST_CASE_NAME} ${anakin_lite_lib_so})
		else()
			target_link_libraries(${TEST_CASE_NAME} ${anakin_lite_lib_static})
		endif()
		#	if(USE_ARM_PLACE)
		#		target_link_libraries(${TEST_CASE_NAME} ${log-lib})
		#	endif()
		set_target_properties(${TEST_CASE_NAME} PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY
				${PROJECT_SOURCE_DIR}/output/unit_test)
	endforeach()
endif()

FILE(GLOB MODEL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

install(DIRECTORY ${ANAKIN_LITE_SABER}/../../saber/lite
		DESTINATION ${PROJECT_SOURCE_DIR}/output/include/saber
		FILES_MATCHING
		PATTERN "*.h")
install(FILES ${ANAKIN_LITE_SABER}/../../saber/saber_types.h
        DESTINATION ${PROJECT_SOURCE_DIR}/output/include/saber)
install(FILES ${MODEL_HEADER}
        DESTINATION ${PROJECT_SOURCE_DIR}/output/include)
install(FILES ${PROJECT_BINARY_DIR}/anakin_config.h
        DESTINATION ${PROJECT_SOURCE_DIR}/output/include)
if(USE_ANDROID)
	install(TARGETS ${anakin_lite_lib_static} ${anakin_lite_lib_so}
		ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/output/lib
		LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/output/lib)
else()
	install(TARGETS ${anakin_lite_lib_static}
			ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/output/lib)
endif()