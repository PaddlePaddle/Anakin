#===============================================================================
# Copyright 2016-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

anakin_find_mlulib()
if (${MLU_FOUND})
    return()
endif()

include(ExternalProject)

set(MLU_PROJECT       "extern_mlu")
set(MLU_SOURCE_DIR    "${ANAKIN_TEMP_THIRD_PARTY_PATH}/mlu")
set(REL_MLU_LIB      "${MLU_SOURCE_DIR}/src/${MLU_PROJECT}/mlu")
set(MLU_INC           "${ANAKIN_THIRD_PARTY_PATH}/mlu/include")
set(MLU_LIB           "${ANAKIN_THIRD_PARTY_PATH}/mlu/lib")
set(MLU_INSTALL_ROOT  ${ANAKIN_THIRD_PARTY_PATH}/mlu)


file(WRITE ${MLU_SOURCE_DIR}/src/install.sh
    "mkdir -p ${MLU_INSTALL_ROOT}/include \n"
    "mkdir -p ${MLU_INSTALL_ROOT}/lib \n"
    "cp ${REL_MLU_LIB}/include/*.h ${MLU_INSTALL_ROOT}/include/ \n" 
    "cp ${REL_MLU_LIB}/lib/*.so ${MLU_INSTALL_ROOT}/lib \n")



ExternalProject_Add(
    ${MLU_PROJECT}
    GIT_REPOSITORY      "xxx"
    GIT_TAG             master 
    PREFIX              ${MLU_SOURCE_DIR}
    INSTALL_COMMAND     sh ${MLU_SOURCE_DIR}/src/install.sh 
)

include_directories(${MLU_INC})
add_library(mlu_lib SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mlu_lib PROPERTY IMPORTED_LOCATION ${MLU_LIB}/libcnrt.so ${MLU_LIB}/libcnml.so)
add_dependencies(mlu_lib ${MLU_PROJECT})
message("mlu lib: ${MLU_LIB}")
list(APPEND ANAKIN_SABER_DEPENDENCIES mlu_lib)
list(APPEND ANAKIN_LINKER_LIBS ${MLU_LIB}/libcnrt.so ${MLU_LIB}/libcnml.so)
