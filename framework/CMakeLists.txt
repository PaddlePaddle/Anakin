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
anakin_fetch_include_recursively(${ANAKIN_SABER})
anakin_fetch_include_recursively(${ANAKIN_MODEL_PARSER})
anakin_fetch_include_recursively(${ANAKIN_UTILS})
anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK}/core)
anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK}/graph)
anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK}/model_parser)
anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK}/operators)
if(BUILD_RPC)
    anakin_fetch_include_recursively(${ANAKIN_SERVICE})
endif()


set(ANAKIN_BASE_SRC "")

# add ak_base_source files
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/c_api "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/core "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/core/net "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/core/operator "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/core/plantform "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/graph "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/graph/llvm "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/graph/llvm/fusion "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/graph/llvm/optimizer "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/operators "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/operators/fusion_ops "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/utils "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/utils/logger "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/utils/unit_test "cpp" ANAKIN_BASE_SRC)

list(APPEND ANAKIN_SRC ${ANAKIN_BASE_SRC})
unset(ANAKIN_BASE_SRC)

# add library to shared or static
if(UNIX OR APPLE)
	if(BUILD_SHARED)
		add_library(${anakin_lib_so} SHARED ${ANAKIN_SRC})
		add_dependencies(${anakin_lib_so} ${ANAKIN_SABER_LIB_TARGET})
		# set shared lib version
		set_target_properties(${anakin_lib_so} PROPERTIES VERSION ${VERSION})
        
		target_link_libraries(${anakin_lib_so} ${ANAKIN_SABER_LIB_TARGET} ${ANAKIN_LINKER_LIBS})
		set_target_properties(${anakin_lib_so} PROPERTIES LINK_FLAGS "")
		set_target_properties(${anakin_lib_so} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/)
        install(DIRECTORY ${ANAKIN_FRAMEWORK} ${ANAKIN_SABER} ${ANAKIN_UTILS}
            	DESTINATION ${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/
            	FILES_MATCHING 
				PATTERN "*.h"
				PATTERN "*.inl")
	endif()
	if(BUILD_STATIC)
		add_library(${anakin_lib_static} STATIC ${ANAKIN_SRC})
		add_dependencies(${anakin_lib_static} ${ANAKIN_SABER_LIB_TARGET})# ${anakin_framework_static})
		#set_target_properties(${anakin_lib_static} PROPERTIES VERSION ${VERSION})
		target_link_libraries(${anakin_lib_static} ${ANAKIN_SABER_LIB_TARGET} ${ANAKIN_LINKER_LIBS})
        set_target_properties(${anakin_lib_static} PROPERTIES LINK_FLAGS "")
		set_target_properties(${anakin_lib_static} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/)
       	install(DIRECTORY ${ANAKIN_FRAMEWORK} ${ANAKIN_SABER} ${ANAKIN_UTILS}
                DESTINATION ${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/
                FILES_MATCHING 
				PATTERN "*.h"
				PATTERN "*.inl")
	endif()
endif()
