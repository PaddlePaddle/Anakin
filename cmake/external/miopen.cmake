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
include(ExternalProject)

set(MIOPEN_PROJECT       extern-miopen)
set(MIOPEN_PREFIX_DIR    ${ANAKIN_THIRD_PARTY_PATH}/miopen)
set(MIOPEN_INSTALL_ROOT  ${ANAKIN_ROOT}/output/miopen)
set(MIOPEN_DOWNLOAD_DIR  ${MIOPEN_PREFIX_DIR})
set(MIOPEN_SOURCE_DIR    ${MIOPEN_PREFIX_DIR})
set(MIOPEN_TMP_DIR       ${ANAKIN_TEMP_THIRD_PARTY_PATH}/${MIOPEN_PROJECT}/tmp)
set(MIOPEN_STAMP_DIR     ${ANAKIN_TEMP_THIRD_PARTY_PATH}/${MIOPEN_PROJECT}/stamp)
set(MIOPEN_BINARY_DIR    ${ANAKIN_TEMP_THIRD_PARTY_PATH}/${MIOPEN_PROJECT}/build)
set(MIOPEN_LIB           ${MIOPEN_INSTALL_ROOT}/lib/libMIOpen.so  CACHE FILEPATH "miopen library." FORCE)

message(STATUS "Scanning external modules ${Green}MIOPEN${ColourReset} ...")

ExternalProject_Add(
    ${MIOPEN_PROJECT}
    GIT_REPOSITORY        https://github.com/ROCmSoftwarePlatform/Anakin.git
    GIT_TAG               anakin-amd_miopen
    PREFIX                ${MIOPEN_PREFIX_DIR}
    DOWNLOAD_DIR          ${MIOPEN_DOWNLOAD_DIR}
    SOURCE_DIR            ${MIOPEN_SOURCE_DIR}
    TMP_DIR               ${MIOPEN_TMP_DIR}
    STAMP_DIR             ${MIOPEN_STAMP_DIR}
    BINARY_DIR            ${MIOPEN_BINARY_DIR}
    CMAKE_ARGS            -DMIOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=${MIOPEN_INSTALL_ROOT} -DCMAKE_INSTALL_LIBDIR=lib
    LOG_BUILD             1
    LOG_INSTALLED         1
    
)

include_directories(${MIOPEN_INSTALL_ROOT}/include)
add_library(miopen SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET miopen PROPERTY IMPORTED_LOCATION ${MIOPEN_LIB})
add_dependencies(miopen ${MIOPEN_PROJECT})

list(APPEND ANAKIN_SABER_DEPENDENCIES miopen)
list(APPEND ANAKIN_LINKER_LIBS ${MIOPEN_LIB})
