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

# ----------------------------------------------------------------------------
# section: help to search src and include files
# ----------------------------------------------------------------------------
# fetch files(.cc .cpp .cu .c or .h .hpp etc.) in dir(search_dir) 
# and save to parent scope var outputs
function(anakin_fetch_files_with_suffix search_dir suffix outputs)
	exec_program(ls ${search_dir}
             ARGS "*.${suffix}"
             OUTPUT_VARIABLE OUTPUT
             RETURN_VALUE VALUE)
	if(NOT VALUE)
		string(REPLACE "\n" ";" OUTPUT_LIST "${OUTPUT}")
		set(abs_dir "")
		foreach(var ${OUTPUT_LIST})
			set(abs_dir ${abs_dir} ${search_dir}/${var})
			#message(STATUS "fetch_result: ${abs_dir}")
		endforeach()
		set(${outputs} ${${outputs}} ${abs_dir} PARENT_SCOPE)
	else()
		#message(WARNING "anakin_fetch_files_recursively ${BoldRed}failed${ColourReset}:\n"
		#                "real_dir:${BoldYellow}${search_dir}${ColourReset}\n"
		#                "suffix:*.${BoldYellow}${suffix}${ColourReset} \n")
	endif()
endfunction()

# recursively fetch files
function(anakin_fetch_files_with_suffix_recursively search_dir suffix outputs)
	file(GLOB_RECURSE ${outputs} ${search_dir} "*.${suffix}")	
	set(${outputs} ${${outputs}} PARENT_SCOPE)
endfunction()

# recursively fetch include dir 
function(anakin_fetch_include_recursively root_dir)
    if (IS_DIRECTORY ${root_dir})
        #message(STATUS "include dir: " ${Magenta}${root_dir}${ColourReset})
		include_directories(${root_dir})
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})                    
            anakin_fetch_include_recursively(${root_dir}/${sub})
        endif()
    endforeach()
endfunction()

# judge fetch files
function(anakin_judge_avx   outputs)
	exec_program(cat /proc/cpuinfo|greps flag|uniq
			OUTPUT_VARIABLE OUTPUT
			RETURN_VALUE VALUE)
	message("it is anakin_judge_avx " OUTPUT)
	set(${outputs} ${OUTPUT} PARENT_SCOPE)
endfunction()

function(anakin_get_cpu_arch   outputs)
	if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
		set(${outputs} native PARENT_SCOPE)
	else()
		exec_program("${CMAKE_CXX_COMPILER} -c -Q -march=native --help=target | grep march | cut -d '=' -f 2 | tr -d '\\040\\011\\012\\015' |cut -d '#' -f 1"
				OUTPUT_VARIABLE OUTPUT
				RETURN_VALUE VALUE)

		set(${outputs} ${OUTPUT} PARENT_SCOPE)
	endif ()
endfunction()
# ----------------------------------------------------------------------------
# section: help to detect the compiler options
# ----------------------------------------------------------------------------
# check and add  compiler options
macro(anakin_check_compiler_flag LANG FLAG RESULT)
    if(NOT DEFINED ${RESULT})
      if("_${LANG}_" MATCHES "_CXX_")
        set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx")
        if("${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror " OR "${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror=unknown-pragmas ")
          FILE(WRITE "${_fname}" "int main() { return 0; }\n")
        else()
          FILE(WRITE "${_fname}" "#pragma\n int main() { return 0; }\n")
        endif()
      elseif("_${LANG}_" MATCHES "_NVCC_")
        set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cu")
        if("${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror " OR "${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror=unknown-pragmas ")
            FILE(WRITE "${_fname}" ""
                        "extern \"C\" __global__ void test() {}\n" 
                        "int main() { return 0; }\n")
        else()
            FILE(WRITE "${_fname}" "#pragma\n"
                        "extern \"C\" __global__ void test() {}\n" 
                        "int main() { return 0; }\n")
        endif()
      else()
        unset(_fname)
      endif()

      if(_fname AND _fname MATCHES "src.cxx")
        MESSAGE(STATUS "Testing ${RESULT}")
        TRY_COMPILE(${RESULT}
          			"${CMAKE_BINARY_DIR}"
          			"${_fname}"
          			COMPILE_DEFINITIONS "${FLAG}" # check the compile option
          			OUTPUT_VARIABLE OUTPUT)
		# result true or false
        if(${RESULT})
            SET(${RESULT} 1 CACHE INTERNAL "Test ${RESULT}")
            MESSAGE(STATUS "Testing ${RESULT} - ${Green}Success${ColourReset}")
        else()
            MESSAGE(STATUS "Testing ${RESULT} - ${Red}Failed${ColourReset}")
            SET(${RESULT} "" CACHE INTERNAL "Test ${RESULT}")
            file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
                "Compilation failed:\n"
                "    source file: '${_fname}'\n"
                "    check option: '${FLAG}'\n"
                "===== BUILD LOG =====\n"
                "${OUTPUT}\n"
                "===== END =====\n\n")
        endif()
      elseif(_fname AND _fname MATCHES "src.cu")
            MESSAGE(STATUS "Testing ${RESULT}")
            EXEC_PROGRAM(nvcc
                         ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/
                         ARGS "${FLAG}" "${_fname}" 
                         OUTPUT_VARIABLE OUTPUT 
                         RETURN_VALUE RET_VALUE)
        if(NOT ${RET_VALUE})
            SET(${RESULT} 1 CACHE INTERNAL "Test ${RESULT}")
            MESSAGE(STATUS "Testing ${RESULT} - ${Cyan}Success${ColourReset}")
        else()
            MESSAGE(STATUS "Testing ${RESULT} - ${Red}Failed${ColourReset}")
            SET(${RESULT} "" CACHE INTERNAL "Test ${RESULT}")
            file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
                "Compilation failed:\n"
                "    source file: '${_fname}'\n"
                "    check option: '${FLAG}'\n"
                "===== BUILD LOG =====\n"
                "${OUTPUT}\n"
                "===== END =====\n\n")
        endif()
      else()
        set(${RESULT} 0)
      endif()
    endif()
endmacro()

macro(anakin_check_flag_support lang flag varname)
	if("_${lang}_" MATCHES "_CXX_")
		set(_lang CXX)
    elseif("_${lang}_" MATCHES "_CU_")
        set(_lang NVCC)
	else()
		set(_lang ${lang})
	endif()
	
	string(TOUPPER "${flag}" ${varname})
	string(REGEX REPLACE "^(/|-)" "HAVE_${_lang}_" ${varname} "${${varname}}")
 	string(REGEX REPLACE " --|-|=| |\\." "_" ${varname} "${${varname}}")

	anakin_check_compiler_flag("${_lang}" "${ARGN} ${flag}" ${${varname}})
endmacro()

macro(anakin_add_compile_option option)
  if(CMAKE_BUILD_TYPE)
    set(CMAKE_TRY_COMPILE_CONFIGURATION ${CMAKE_BUILD_TYPE})
  endif()
  if("_${ARGV1}_" MATCHES "_NVCC_")
    anakin_check_flag_support(CU "${option}" _varname)
    if(${_varname})
        set(ANAKIN_NVCC_FLAG "${ANAKIN_NVCC_FLAG} ${option}")
    endif()
  else()
    anakin_check_flag_support(CXX "${option}" _varname)
    if(${_varname})
        set(ANAKIN_EXTRA_CXX_FLAGS "${ANAKIN_EXTRA_CXX_FLAGS} ${option}")
    endif()
  endif()
endmacro()

# ----------------------------------------------------------------------------
# section: Provides an anakin config option macro
# usage：  anakin_option(var "help string to describe the var" [if or IF (condition)])
# ----------------------------------------------------------------------------
macro(anakin_option variable description value)
	set(__value ${value})
 	set(__condition "")
 	set(__varname "__value")
	foreach(arg ${ARGN})
 		if(arg STREQUAL "IF" OR arg STREQUAL "if")
 			set(__varname "__condition")
 		else()
 			list(APPEND ${__varname} ${arg})
 		endif()
 	endforeach()
 	unset(__varname)
 	if(__condition STREQUAL "")
 		set(__condition 2 GREATER 1)
 	endif()
	
	if(${__condition})
		if(__value MATCHES ";")
 			if(${__value})
 				option(${variable} "${description}" ON)
 			else()
 				option(${variable} "${description}" OFF)
 			endif()
 		elseif(DEFINED ${__value})
			if(${__value})
 				option(${variable} "${description}" ON)
			else()
				option(${variable} "${description}" OFF)
 			endif()
 		else()
 			option(${variable} "${description}" ${__value})
		endif()
	else()
 		unset(${variable} CACHE)
 	endif()
 	unset(__condition)
 	unset(__value)
 endmacro()

# ----------------------------------------------------------------------------
# section: Provides functions for an anakin opencl kernel generator
# ----------------------------------------------------------------------------
function(anakin_generate_kernel anakin_root_dir)
	set(kerel_generate_script_path ${anakin_root_dir}/src/cl_kernels)
	exec_program(${kerel_generate_script_path}/generate.sh
				ARGS " ${anakin_root_dir}"
             	OUTPUT_VARIABLE OUTPUT
             	RETURN_VALUE VALUE)
	if(NOT VALUE)	
	    message(STATUS "generate kernel files ${Green}${OUTPUT}${ColourReset} successfully.")
	else()
	    message(FATAL_ERROR "anakin_generate_kernel\npath: ${kerel_generate_script_path}\nscript: generate.sh ")
	endif()
endfunction()


# ----------------------------------------------------------------------------
# section: generate the protobuf .h and .cpp files.
# ----------------------------------------------------------------------------
function(anakin_gen_pb proto_src_path)
    set(__working_dir ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PROTO_TEMP/)
    foreach(__proto_file ${ARGN}) 
        exec_program(${PROTOBUF_PROTOC_EXECUTABLE} ${__working_dir} ARGS " -I=${proto_src_path} --cpp_out=. ${__proto_file}" 
                                              OUTPUT_VARIABLE OUTPUT RETURN_VALUE VALUE)
        if(NOT VALUE)
            anakin_fetch_files_with_suffix(${__working_dir} "h" PROTO_GENERATE_H)
            # get *.cpp or *.cc
            anakin_fetch_files_with_suffix(${__working_dir} "c*" PROTO_GENERATE_C)
            foreach(__include_file ${PROTO_GENERATE_H})
                exec_program(mv ARGS ${__include_file} ${proto_src_path} 
                                OUTPUT_VARIABLE __out RETURN_VALUE __value)
            endforeach()
            foreach(__src_file ${PROTO_GENERATE_C})
                if(POLICY CMP0007) 
                    cmake_policy(PUSH) 
                    cmake_policy(SET CMP0007 NEW) 
                endif()
                string(REPLACE "." ";" SRC_LIST ${__src_file})
                list(GET SRC_LIST -1 __src_file_name_suffix)
				list(GET SRC_LIST -3 __src_file_name)

				string(REPLACE "/" ";" SRC_LIST_PATH ${__src_file_name})
				list(GET SRC_LIST_PATH -1 __pure_src_file_name)

				if(__src_file_name_suffix EQUAL "cpp")
					set(__full_src_filename "${__pure_src_file_name}.pb.cpp")
				else()
					set(__full_src_filename "${__pure_src_file_name}.pb.cc")
				endif()
				exec_program(mv ARGS " ${__working_dir}${__full_src_filename}  ${proto_src_path}/${__pure_src_file_name}.pb.cpp" 
								OUTPUT_VARIABLE __out
								RETURN_VALUE __value)
				if(POLICY CMP0007)
  					cmake_policy(POP)
				endif()
            endforeach()
        else()
            message(FATAL_ERROR "anakin_gen_bp: ${__file} \n error msg: ${OUTPUT}")
        endif()
    endforeach()
endfunction()

function(anakin_protos_processing)
	set(PROTO_SRC_PATH ${ANAKIN_MODEL_PARSER}/proto)
    set(SERVICE_API_SRC_PATH ${ANAKIN_SERVICE}/api)

	set(__working_dir ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PROTO_TEMP/)
	
	anakin_fetch_files_with_suffix(${PROTO_SRC_PATH} "proto" PROTO_SRC_FILES)
    anakin_fetch_files_with_suffix(${SERVICE_API_SRC_PATH} "proto" SERVICE_API_PROTO_SRC_FILES)
    anakin_gen_pb(${PROTO_SRC_PATH} ${PROTO_SRC_FILES})
    if(BUILD_RPC)
        anakin_gen_pb(${SERVICE_API_SRC_PATH} ${SERVICE_API_PROTO_SRC_FILES})
    endif()
endfunction()

# ----------------------------------------------------------------------------
# section: Provides macro for an anakin warning diasable
# ----------------------------------------------------------------------------
macro(anakin_disable_warnings)
	set(__flag_vars "")

  	foreach(arg ${ARGN})
    	if(arg MATCHES "^CMAKE_")
      		list(APPEND __flag_vars ${arg})
		endif()
  	endforeach()

  	if(NOT __flag_vars)
    	set(__flag_vars CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  	endif()

  	if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)
    	foreach(var ${__flag_vars})
			string(REPLACE " " ";"  __list_flag  ${${var}})	
			foreach(warning ${__list_flag})
				if(NOT warning MATCHES "^-Wno-")
					if((warning MATCHES "^-W") AND (NOT warning STREQUAL "-W"))
						string(REGEX REPLACE "${warning}" "" ${var} "${${var}}")
					endif()
					# this may lead to error in project.
					if(warning STREQUAL "-fpermissive")
                        #string(REGEX REPLACE "${warning}" "" ${var} "${${var}}")
					endif()
				endif()
			endforeach()
    	endforeach()
  	endif()
	unset(_flag_vars)
	unset(_gxx_warnings)
endmacro()

# ----------------------------------------------------------------------------
# section: Get file name in path (without suffix)
# ----------------------------------------------------------------------------
macro(anakin_get_file_name path file_name)
    string(REPLACE "/" ";" split_code_list ${${path}})
    list(GET split_code_list -1 real_code_with_suffix) 
    string(REPLACE "." ";" split_code_list ${real_code_with_suffix})
    list(GET split_code_list 0 real_code_name)
    set(${file_name} ${real_code_name})
    unset(split_code_list)
    unset(real_code_name)
    unset(real_code_with_suffix)
endmacro()

macro(anakin_set_upscope src)
    set(${src} ${${src}} PARENT_SCOPE)
endmacro()

