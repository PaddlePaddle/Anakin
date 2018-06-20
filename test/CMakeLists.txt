# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# Modifications (c) 2018 Advanced Micro Devices, Inc.
# @file     CMakeLists files in the unit test directory of project
# @auther   cuichaowen
# @date     2017-10-24
# ----------------------------------------------------------------------------
# used for temporary
if(USE_ANAKIN_FRAMEWORK)
    anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK})
    anakin_fetch_include_recursively(${ANAKIN_MODEL_PARSER})
endif()
anakin_fetch_include_recursively(${ANAKIN_SABER})

if(USE_GTEST)
    anakin_find_gtest()
endif()

if(USE_ANAKIN_FRAMEWORK)
anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/core "cpp" ANAKIN_TEST_CASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/graph "cpp" ANAKIN_TEST_CASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/operators "cpp" ANAKIN_TEST_CASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/net "cpp" ANAKIN_TEST_CASE_SRC)
endif()

if(NVIDIA_GPU)
anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/saber/cuda "cpp" ANAKIN_TEST_CASE_SRC)
endif()

if(USE_X86_PLACE)
anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/saber/x86 "cpp" ANAKIN_TEST_CASE_SRC)
endif()

if(USE_AMD)
    anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/saber/amd "cpp" ANAKIN_TEST_CASE_SRC)
endif()

if(USE_ARM_PLACE) #build unit test for arm devices
	anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/saber/arm "cpp" ANAKIN_TEST_CASE_SRC)
	if(USE_OPENMP)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	endif()
endif()

file(REMOVE ${PROJECT_SOURCE_DIR}/output/unit_test/*)

# build test cases
foreach(SRC_NAME ${ANAKIN_TEST_CASE_SRC})
	#unpack the dir "/"
	string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
	list(GET SEXY_LIST -1 TEST_CASE_NAME)
	#get the file name without suffix
	string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
	list(GET SEXY_LIST 0 TEST_CASE_NAME)
	add_executable(${TEST_CASE_NAME}  ${SRC_NAME})
        
        if(USE_ANAKIN_FRAMEWORK)
            if(BUILD_SHARED)
                target_link_libraries(${TEST_CASE_NAME} ${anakin_lib_so})
            else()
                target_link_libraries(${TEST_CASE_NAME} ${anakin_lib_static})
            endif()
        else()
            message(STATUS "link ${ANAKIN_SABER_LIB_TARGET}")
            target_link_libraries(${TEST_CASE_NAME} ${ANAKIN_LINKER_LIBS} ${ANAKIN_SABER_LIB_TARGET})
        endif()

	set_target_properties(${TEST_CASE_NAME} PROPERTIES
						RUNTIME_OUTPUT_DIRECTORY 
						${PROJECT_SOURCE_DIR}/output/unit_test)
endforeach()
