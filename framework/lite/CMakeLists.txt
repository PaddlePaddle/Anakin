# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# ----------------------------------------------------------------------------

# used for temporary
anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK})
anakin_fetch_include_recursively(${ANAKIN_SABER})
anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK}/lite)

#if(USE_ARM_PLACE)
#	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/lite/core "cpp" ANAKIN_SABER_ARM_LITE_SRC)
#	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/lite/funcs "cpp" ANAKIN_SABER_ARM_LITE_SRC)
#	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/lite/funcs/neon "cpp" ANAKIN_SABER_ARM_LITE_SRC)
#	anakin_fetch_files_with_suffix(${ANAKIN_SABER}/lite/funcs/neon/impl "cpp" ANAKIN_SABER_ARM_LITE_SRC)
#
#	set(anakin_saber_arm_lite_static "anakin_saber_arm_lite_static")
#	add_library(${anakin_saber_arm_lite_static} STATIC ${ANAKIN_SABER_ARM_LITE_SRC})
#	set_target_properties(${anakin_saber_arm_lite_static} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/)
#	install(DIRECTORY ${ANAKIN_SABER}/lite
#        	DESTINATION ${PROJECT_SOURCE_DIR}/tools/anakin-lite
#        	FILES_MATCHING 
#			PATTERN "*.h"
#			PATTERN "*.inl")
#endif()

# install saber arm lite to tools/anakin-lite
#install(FILES ${ANAKIN_SABER}/saber_types.h DESTINATION ${PROJECT_SOURCE_DIR}/tools/anakin-lite/saber)
#install(DIRECTORY ${ANAKIN_SABER}/lite DESTINATION ${PROJECT_SOURCE_DIR}/tools/anakin-lite/saber)
#install(DIRECTORY ${ANAKIN_UNIT_TEST}/lite DESTINATION ${PROJECT_SOURCE_DIR}/tools/anakin-lite/test)
#install(DIRECTORY ${ANAKIN_UTILS}/unit_test DESTINATION ${PROJECT_SOURCE_DIR}/tools/anakin-lite/utils)
#install(DIRECTORY ${ANAKIN_UTILS}/logger DESTINATION ${PROJECT_SOURCE_DIR}/tools/anakin-lite/utils)

anakin_fetch_files_with_suffix(${ANAKIN_LITE_FRAMEWORK}/generator/src "cpp" ANAKIN_LITE_EXE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_LITE_FRAMEWORK} "cpp" ANAKIN_LITE_SRC)

# build test cases
foreach(SRC_NAME ${ANAKIN_LITE_EXE_SRC})
	#unpack the dir "/"
	string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
	list(GET SEXY_LIST -1 LITE_EXE_NAME)
	#get the file name without suffix
	string(REPLACE "." ";" SEXY_LIST ${LITE_EXE_NAME})
    list(GET SEXY_LIST 0 LITE_EXE_NAME)
	add_executable(${LITE_EXE_NAME}  ${SRC_NAME} ${ANAKIN_LITE_SRC})
    if(BUILD_SHARED) 
        target_link_libraries(${LITE_EXE_NAME} ${anakin_lib_so})
    else()
        target_link_libraries(${LITE_EXE_NAME} ${anakin_lib_static})
    endif()	
	set_target_properties(${LITE_EXE_NAME} PROPERTIES
						  RUNTIME_OUTPUT_DIRECTORY 
						  ${PROJECT_SOURCE_DIR}/output/generator/src)
	install(DIRECTORY ${ANAKIN_LITE_FRAMEWORK}/generator
			DESTINATION ${PROJECT_SOURCE_DIR}/output/
			FILES_MATCHING
			PATTERN "*.sh"
			PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
			GROUP_EXECUTE GROUP_READ)
endforeach()
