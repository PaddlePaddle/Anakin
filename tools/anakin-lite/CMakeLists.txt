# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# ----------------------------------------------------------------------------

# used for temporary
anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK})
anakin_fetch_include_recursively(${ANAKIN_SABER})

anakin_fetch_files_with_suffix(${ANAKIN_LITE_EXE}/src "cpp" ANAKIN_LITE_EXE_SRC)

# build test cases
foreach(SRC_NAME ${ANAKIN_LITE_EXE_SRC})
	#unpack the dir "/"
	string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
	list(GET SEXY_LIST -1 LITE_EXE_NAME)
	#get the file name without suffix
	string(REPLACE "." ";" SEXY_LIST ${LITE_EXE_NAME})
    list(GET SEXY_LIST 0 LITE_EXE_NAME)
    add_executable(${LITE_EXE_NAME}  ${SRC_NAME})
    if(BUILD_SHARED) 
        target_link_libraries(${LITE_EXE_NAME} ${anakin_lib_so})
    else()
        target_link_libraries(${LITE_EXE_NAME} ${anakin_lib_static})
    endif()	
	set_target_properties(${LITE_EXE_NAME} PROPERTIES
						  RUNTIME_OUTPUT_DIRECTORY 
						  ${ANAKIN_LITE_EXE}/bin/)
endforeach()
