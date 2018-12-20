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
anakin_fetch_files_with_suffix(${ANAKIN_MODEL_PARSER}/proto "proto" ANAKIN_PROTO_SRC)

set(PROTOC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/model_parser/proto")
file(MAKE_DIRECTORY ${PROTOC_OUT_DIR})
include_directories(${PROTOC_OUT_DIR})

set(ANAKIN_BASE_SRC "")

if(USE_NANOPB)
  include_directories(${NANOPB_DIR})

  anakin_fetch_files_with_suffix(${ANAKIN_MODEL_PARSER}/parser/nanopb/ "cpp" ANAKIN_BASE_SRC)

  add_definitions(-DPB_FIELD_16BIT)

  add_custom_command(
    OUTPUT  "${PROTOBUF_PROTOC_EXECUTABLE}" "${NANOPB_DIR}/pb_decode.c" "${NANOPB_DIR}/pb_common.c"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${NANOPB_DIR}
    COMMAND bash ARGS -c "wget -qO- ${NANOPB_DOWNLOAD_URL} | tar xz -C ${NANOPB_DIR} --strip 1"
    COMMENT "Downlaoding prebuilt nanopb-${NANOPB_VERSION}..."
    VERBATIM)

  set(PROTOC_OUT_ARGS "--nanopb_out=-I${ANAKIN_MODEL_PARSER}/proto:")

  list(APPEND ANAKIN_SRC "${NANOPB_DIR}/pb_decode.c" "${NANOPB_DIR}/pb_common.c")
else()
  set(PROTOC_OUT_ARGS "--cpp_out=")
endif()

foreach(__file ${ANAKIN_PROTO_SRC})
  get_filename_component(__file_name ${__file} NAME_WE)
  if(USE_NANOPB)
    set(__out_src_name "${PROTOC_OUT_DIR}/${__file_name}.pb.c")
    if(EXISTS "${ANAKIN_MODEL_PARSER}/proto/${__file_name}.options")
      set(__proto_options "${ANAKIN_MODEL_PARSER}/proto/${__file_name}.options")
    endif()
  else()
    set(__out_src_name "${PROTOC_OUT_DIR}/${__file_name}.pb.cc")
  endif()
  set(__out_header_name "${PROTOC_OUT_DIR}/${__file_name}.pb.h")
  add_custom_command(
    OUTPUT "${__out_src_name}" "${__out_header_name}"
    COMMAND "${PROTOBUF_PROTOC_EXECUTABLE}"
    ARGS "-I${ANAKIN_MODEL_PARSER}/proto" ${__file} "${PROTOC_OUT_ARGS}${PROTOC_OUT_DIR}"
    DEPENDS ${PROTOBUF_PROTOC_EXECUTABLE} ${__file} ${__proto_options}
    COMMENT "Compiling ${__file_name}.proto using ${PROTOBUF_PROTOC_EXECUTABLE}...")
  list(APPEND ANAKIN_SRC "${__out_src_name}")
endforeach()

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
anakin_fetch_files_with_suffix(${ANAKIN_FRAMEWORK}/model_parser/parser "cpp" ANAKIN_BASE_SRC)

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
    target_link_libraries(${anakin_lib_static} ${ANAKIN_SABER_LIB_TARGET} ${ANAKIN_LINKER_LIBS})
    if(USE_SGX)
      target_link_libraries(${anakin_lib_static} ${SGX_CONFIG_INTERFACE})
    endif()
    set_target_properties(${anakin_lib_static} PROPERTIES LINK_FLAGS "")
    set_target_properties(${anakin_lib_static} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/)
    install(DIRECTORY ${ANAKIN_FRAMEWORK} ${ANAKIN_SABER} ${ANAKIN_UTILS}
      DESTINATION ${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/
      FILES_MATCHING 
      PATTERN "*.h"
      PATTERN "*.inl")
  endif()
endif()
