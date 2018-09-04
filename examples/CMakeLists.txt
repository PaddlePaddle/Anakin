# used for temporary
anakin_fetch_include_recursively(${ANAKIN_FRAMEWORK})
anakin_fetch_include_recursively(${ANAKIN_MODEL_PARSER})
anakin_fetch_include_recursively(${ANAKIN_SABER})

if(NVIDIA_GPU)
anakin_fetch_files_with_suffix(${ANAKIN_EXAMPLES}/cuda "cpp" ANAKIN_TEST_CASE_SRC)
endif()

if(AMD_GPU)
anakin_fetch_files_with_suffix(${ANAKIN_EXAMPLES}/amd "cpp" ANAKIN_TEST_CASE_SRC)
endif()

if(USE_X86_PLACE)
anakin_fetch_files_with_suffix(${ANAKIN_EXAMPLES}/x86 "cpp" ANAKIN_TEST_CASE_SRC)
endif()

if(USE_ARM_PLACE) #build unit test for arm devices
	anakin_fetch_files_with_suffix(${ANAKIN_EXAMPLES}/arm "cpp" ANAKIN_TEST_CASE_SRC)
	if(USE_OPENMP)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	endif()
	if (USE_PROTOBUF)
		find_library(log-lib log)
	endif()
endif()

file(REMOVE ${PROJECT_SOURCE_DIR}/output/examples/*)

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
		target_link_libraries(${TEST_CASE_NAME} ${anakin_lib_so} ${ANAKIN_LINKER_LIBS})
	else()
		target_link_libraries(${TEST_CASE_NAME} -Wl,--whole-archive ${anakin_lib_static} -Wl,--no-whole-archive ${ANAKIN_LINKER_LIBS})
	endif()
	if(USE_ARM_PLACE)
		target_link_libraries(${TEST_CASE_NAME} ${log-lib})
	endif()
    if(USE_OPENCV)
        if (USE_ARM_PLACE)
            target_link_libraries(${TEST_CASE_NAME} -lopencv_core -lopencv_highgui -lopencv_imgproc
					-ltbb -llibtiff -llibpng -llibjpeg -llibjasper -lIlmImf -lc -lz -llog -ldl)
        else()
            target_link_libraries(${TEST_CASE_NAME} -lopencv_core -lopencv_highgui -lopencv_imgproc)
        endif()
    endif()
	set_target_properties(${TEST_CASE_NAME} PROPERTIES
						RUNTIME_OUTPUT_DIRECTORY 
						${PROJECT_SOURCE_DIR}/output/examples)
endforeach()
