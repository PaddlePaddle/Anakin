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
option(USE_OPENCV "use opencv in unit test" NO)
option(ENABLE_OP_TIMER "get time consumption of each op" NO)
option(USE_ANDROID_LOG "use build-in android logger" NO)

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
	set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/" CACHE PATH "Installation Directory")
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
	add_compile_options(-Ofast)
	add_compile_options(-ffast-math)
	add_compile_options(-Os)
	add_compile_options(-DNDEBUG)
endif()

set(anakin_lite_lib_so "anakin_lite")
set(anakin_lite_lib_static "anakin_lite_static")

# disable shared library on xcode ios
if(USE_IOS)
	set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
endif()
add_compile_options(-std=c++11)
add_compile_options(-fPIC)
if (USE_ANDROID)
	#add_compile_options(-ffunction-sections)
	#add_compile_options(-fdata-sections)
	#add_compile_options(-fvisibility=hidden)
	#add_compile_options(-fvisibility-inlines-hidden)
	add_compile_options(-ldl)
	add_compile_options(-Os)
	add_compile_options(-Ofast)
	if(USE_ARMV8)
		set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
		set(MAKE_STATIC_LINKER_FLAGS "${MAKE_STATIC_LINKER_FLAGS} -Wl,--gc-sections")
	else()
		set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe")
		set(MAKE_STATIC_LINKER_FLAGS "${MAKE_STATIC_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe")
	endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

#anakin_add_compile_option(-fstrict-aliasing)
add_compile_options(-W)
add_compile_options(-Wall)
add_compile_options(-Wno-unused-variable) # no unused-variable
add_compile_options(-Wformat)
add_compile_options(-Wmissing-declarations)
add_compile_options(-Winit-self)
add_compile_options(-Wpointer-arith)
add_compile_options(-Wno-shadow)
add_compile_options(-fpermissive)
add_compile_options(-Wsign-promo)
add_compile_options(-fdiagnostics-show-option)
add_compile_options(-Wno-undef)
add_compile_options(-Wno-narrowing)
add_compile_options(-Wno-unknown-pragmas)
add_compile_options(-Wno-delete-non-virtual-dtor)
add_compile_options(-Wno-comment)
add_compile_options(-Wno-sign-compare)
add_compile_options(-Wno-ignored-qualifiers)
add_compile_options(-Wno-enum-compare)
add_compile_options(-Wno-unused-parameter)
add_compile_options(-Wno-unused-function)

if(USE_ANDROID)
	add_compile_options(-pie)
	if(USE_ARMV8)
	else()
		add_compile_options(-mfloat-abi=softfp)
		add_compile_options(-mfpu=neon)
	endif()
	add_compile_options(-ffast-math)
	add_compile_options(-lc)
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
FILE(GLOB BUILD_SRC_FILES6 "${ANAKIN_LITE_SABER}/net/*.cpp")
FILE(GLOB BUILD_SRC_FILES7 "${ANAKIN_LITE_SABER}/utils/*.cpp")
FILE(GLOB HEADER_NET "${ANAKIN_LITE_SABER}/net/*.h")
FILE(GLOB HEADER_UTILS "${ANAKIN_LITE_SABER}/utils/*.h")

if(USE_ANDROID)
	FILE(GLOB UNIT_TEST_LITE_SRC "${UNIT_TEST_LITE}/*.cpp")
endif()

if(USE_ANDROID_LOG)
	find_library(log-lib log)
endif()

add_library(ANAKIN_LITE_OBJS OBJECT ${BUILD_SRC_FILES1} ${BUILD_SRC_FILES2} ${BUILD_SRC_FILES3} ${BUILD_SRC_FILES4} ${BUILD_SRC_FILES5}
		${BUILD_SRC_FILES6} ${HEADER_NET}
		${BUILD_SRC_FILES7} ${HEADER_UTILS}
		)

if(USE_ANDROID)
	add_library(${anakin_lite_lib_so} SHARED $<TARGET_OBJECTS:ANAKIN_LITE_OBJS>)
	set_target_properties(${anakin_lite_lib_so} PROPERTIES
					  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/)
	target_link_libraries(${anakin_lite_lib_so} ${log-lib})
endif()
add_library(${anakin_lite_lib_static} STATIC $<TARGET_OBJECTS:ANAKIN_LITE_OBJS>)
set_target_properties(${anakin_lite_lib_static} PROPERTIES
					  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/)

if(USE_OPENCV)
	# set your opencv path here
	# for android
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/arm-android/opencv/include/)
	LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/arm-android/opencv/lib/armeabi-v7a/)
endif()

if(BUILD_LITE_UNIT_TEST)
	add_compile_options(-fexceptions)
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
			target_link_libraries(${TEST_CASE_NAME} -Wl,--whole-archive ${anakin_lite_lib_static} -Wl,--no-whole-archive)
		endif()
		if(USE_OPENCV)
			target_link_libraries(${TEST_CASE_NAME} -lopencv_core -lopencv_highgui -lopencv_imgproc
						-ltbb -llibtiff -llibpng -llibjpeg -llibjasper -lIlmImf -lc -lz -llog -ldl)
		endif()
		if(USE_ANDROID_LOG)
			target_link_libraries(${TEST_CASE_NAME} ${log-lib})
		endif()
		set_target_properties(${TEST_CASE_NAME} PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY
				${CMAKE_INSTALL_PREFIX}/unit_test)
	endforeach()
endif()

FILE(GLOB MODEL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

install(DIRECTORY ${ANAKIN_LITE_SABER}/../../saber/lite/core
		DESTINATION ${CMAKE_INSTALL_PREFIX}/include/saber/lite
		FILES_MATCHING
		PATTERN "*.h")
install(DIRECTORY ${ANAKIN_LITE_SABER}/../../saber/lite/net
		DESTINATION ${CMAKE_INSTALL_PREFIX}/include/saber/lite
		FILES_MATCHING
		PATTERN "*.h")
install(DIRECTORY ${ANAKIN_LITE_SABER}/../../saber/lite/utils
		DESTINATION ${CMAKE_INSTALL_PREFIX}/include/saber/lite
		FILES_MATCHING
		PATTERN "*.h")
install(FILES ${ANAKIN_LITE_SABER}/../../saber/lite/funcs/timer_lite.h
		DESTINATION ${CMAKE_INSTALL_PREFIX}/include/saber/lite/funcs)
install(FILES ${ANAKIN_LITE_SABER}/../../saber/lite/funcs/op_base.h
		DESTINATION ${CMAKE_INSTALL_PREFIX}/include/saber/lite/funcs)
install(FILES ${ANAKIN_LITE_SABER}/../../saber/lite/funcs/op_param.h
		DESTINATION ${CMAKE_INSTALL_PREFIX}/include/saber/lite/funcs)
install(FILES ${ANAKIN_LITE_SABER}/../../saber/saber_types.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/saber)
install(FILES ${MODEL_HEADER}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES ${PROJECT_BINARY_DIR}/anakin_config.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

install(TARGETS ${anakin_lite_lib_static}
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(USE_ANDROID)
	install(TARGETS ${anakin_lite_lib_static} ${anakin_lite_lib_so}
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
