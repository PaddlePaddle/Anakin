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
# limitation under the License.

# used for temporary
if(BUILD_WITH_FRAMEWORK)
    anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK})
    anakin_fetch_include_recursively(${ANAKIN_MODEL_PARSER})
endif()

anakin_fetch_include_recursively(${ANAKIN_SABER})

if(USE_GTEST)
    anakin_find_gtest()
endif()

if(BUILD_WITH_FRAMEWORK)
    anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/core "cpp" ANAKIN_TEST_CASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/graph "cpp" ANAKIN_TEST_CASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/operators "cpp" ANAKIN_TEST_CASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/net "cpp" ANAKIN_TEST_CASE_SRC)
    anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/lite "cpp" ANAKIN_TEST_CASE_SRC)
    if(BUILD_RPC)
        anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/framework/service "cpp" ANAKIN_TEST_CASE_SRC)
    endif()
endif()

anakin_fetch_files_with_suffix(${ANAKIN_UNIT_TEST}/saber "cpp" ANAKIN_TEST_CASE_SRC)

file(REMOVE ${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/unit_test/*)

install(FILES ${PROJECT_BINARY_DIR}/anakin_config.h
        DESTINATION ${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/include)

# build test cases
foreach(SRC_NAME ${ANAKIN_TEST_CASE_SRC})
	#unpack the dir "/"
	string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
	list(GET SEXY_LIST -1 TEST_CASE_NAME)
	#get the file name without suffix
	string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
	list(GET SEXY_LIST 0 TEST_CASE_NAME)
	add_executable(${TEST_CASE_NAME}  ${SRC_NAME})
	if(BUILD_SHARED)
            if(BUILD_WITH_FRAMEWORK)
                target_link_libraries(${TEST_CASE_NAME} ${anakin_lib_so} ${ANAKIN_LINKER_LIBS})
            else()
                target_link_libraries(${TEST_CASE_NAME} ${ANAKIN_SABER_LIB_TARGET})
            endif()
	else()
        if(BUILD_WITH_FRAMEWORK) 
            target_link_libraries(${TEST_CASE_NAME} -Wl,--whole-archive ${anakin_lib_static} -Wl,--no-whole-archive)
        else() 
            target_link_libraries(${TEST_CASE_NAME} -Wl,--whole-archive ${ANAKIN_SABER_LIB_TARGET} -Wl,--no-whole-archive)
        endif()

	endif()
	set_target_properties(${TEST_CASE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
						  ${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/unit_test)
endforeach()
