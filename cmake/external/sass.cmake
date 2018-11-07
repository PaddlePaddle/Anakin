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

if (EXISTS ${ANAKIN_THIRD_PARTY_PATH}/sass/lib/sm_50 
        AND EXISTS ${ANAKIN_THIRD_PARTY_PATH}/sass/lib/sm_61)
    include_directories(${ANAKIN_THIRD_PARTY_PATH}/sass/include)
    return() 
endif()  

include(ExternalProject)

set(SASS_PROJECT       "extern_sass")
set(SASS_SOURCE_DIR    "${ANAKIN_TEMP_THIRD_PARTY_PATH}/sass")
set(REAL_SASS_SRC      "${SASS_SOURCE_DIR}/src/${SASS_PROJECT}")
set(SASS_INC           "${ANAKIN_THIRD_PARTY_PATH}/sass/include")
set(SASS_LIB           "${ANAKIN_THIRD_PARTY_PATH}/sass/lib")
set(SASS_INSTALL_ROOT  ${ANAKIN_THIRD_PARTY_PATH}/sass)

include_directories(${SASS_INC})

file(WRITE ${SASS_SOURCE_DIR}/src/build.sh 
    "cmake ../${SASS_PROJECT} -DSELECT_ARCH=61;make -j$(nproc) \n"
    "cmake ../${SASS_PROJECT} -DSELECT_ARCH=50;make -j$(nproc) \n")

file(WRITE ${SASS_SOURCE_DIR}/src/install.sh
    "mkdir -p ${SASS_INSTALL_ROOT}/include \n"
    "mkdir -p ${SASS_INSTALL_ROOT}/lib/sm_50 \n"
    "mkdir -p ${SASS_INSTALL_ROOT}/lib/sm_61 \n"
    "cp ${REAL_SASS_SRC}/nv/*.h ${SASS_INSTALL_ROOT}/include/ \n" 
    "cp *50.a ${SASS_INSTALL_ROOT}/lib/sm_50 \n"
    "cp *61.a ${SASS_INSTALL_ROOT}/lib/sm_61 \n")



ExternalProject_Add(
    ${SASS_PROJECT}
    GIT_REPOSITORY      "ssh://git@icode.baidu.com:8235/baidu/sys-hic-gpu/anakin_saber_lib"
    PREFIX              ${SASS_SOURCE_DIR}
    BUILD_COMMAND       sh ${SASS_SOURCE_DIR}/src/build.sh 
    INSTALL_COMMAND     sh ${SASS_SOURCE_DIR}/src/install.sh 
)

add_library(sass_lib SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET sass_lib PROPERTY IMPORTED_LOCATION ${SASS_LIB})
add_dependencies(sass_lib ${SASS_PROJECT})

list(APPEND ANAKIN_SABER_DEPENDENCIES sass_lib)
