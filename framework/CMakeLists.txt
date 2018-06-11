# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# @file     CMakeLists files in the framework directory of project
# @auther   cuichaowen
# @date     2017-10-24
# ----------------------------------------------------------------------------
anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK})
anakin_fetch_include_recursively(${ANAKIN_SABER})
anakin_fetch_include_recursively(${ANAKIN_MODEL_PARSER})
anakin_fetch_include_recursively(${ANAKIN_UTILS})

set(ANAKIN_BASE_SRC "")

# add ak_base_source files
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
		set_target_properties(${anakin_lib_so} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/)
        install(DIRECTORY ${ANAKIN_FRAMEWORK} ${ANAKIN_SABER} ${ANAKIN_UTILS}
            	DESTINATION ${PROJECT_SOURCE_DIR}/output/
            	FILES_MATCHING 
				PATTERN "*.h"
				PATTERN "*.inl")
	endif()
	if(BUILD_STATIC)
		#add_compile_options("--whole-archive")
        #set(anakin_framework_static "framework_lib")
        #add_library(${anakin_framework_static} STATIC ${ANAKIN_SRC})
		add_library(${anakin_lib_static} STATIC ${ANAKIN_SRC})
		add_dependencies(${anakin_lib_static} ${ANAKIN_SABER_LIB_TARGET})# ${anakin_framework_static})
		#set_target_properties(${anakin_lib_static} PROPERTIES VERSION ${VERSION})
		target_link_libraries(${anakin_lib_static} ${ANAKIN_SABER_LIB_TARGET} ${ANAKIN_LINKER_LIBS})
        set_target_properties(${anakin_lib_static} PROPERTIES LINK_FLAGS "")
		set_target_properties(${anakin_lib_static} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/)
        install(DIRECTORY ${ANAKIN_FRAMEWORK} ${ANAKIN_SABER} ${ANAKIN_UTILS}
                	DESTINATION ${PROJECT_SOURCE_DIR}/output/
                	FILES_MATCHING 
					PATTERN "*.h"
					PATTERN "*.inl")
	endif()
endif()
