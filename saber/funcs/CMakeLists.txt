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

aux_source_directory(. DIR_BASE_SRCS)
aux_source_directory(impl DIR_BASE_SRCS_IMPL)
if(USE_ARM)
    aux_source_directory(impl/arm ARM_BASE_SRCS)
endif()
if(USE_X86)
    aux_source_directory(impl/x86 X86_BASE_SRCS)
endif()
if(USE_CUDA)
    aux_source_directory(impl/cuda CUDA_BASE_SRCS)
endif()
if(AMD_GPU)
    aux_source_directory(impl/amd AMD_BASE_SRCS)
endif()

if(USE_BM_PLACE)
    aux_source_directory(impl/bm BM_BASE_SRCS)
endif()

set(DIR_SRCS_CUR "")
foreach(SRC_NAME ${DIR_BASE_SRCS})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

foreach(SRC_NAME ${DIR_BASE_SRCS_IMPL})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

foreach(SRC_NAME ${ARM_BASE_SRCS})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

foreach(SRC_NAME ${CUDA_BASE_SRCS})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

foreach(SRC_NAME ${X86_BASE_SRCS})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()


foreach(SRC_NAME ${AMD_BASE_SRCS})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

foreach(SRC_NAME ${BM_BASE_SRCS})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

#message(STATUS "base source files:${DIR_SRCS_CUR}")

#set(CUDA_SRC "${CUDA_SRC}${CUDA_BASE_SRCS};" PARENT_SCOPE)

set(DIRSRCS "${DIRSRCS}${DIR_SRCS_CUR};" PARENT_SCOPE)

