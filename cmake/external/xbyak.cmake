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

include(ExternalProject)

set(XBYAK_PROJECT       extern_xbyak)
set(XBYAK_PREFIX_DIR    ${ANAKIN_TEMP_THIRD_PARTY_PATH}/xbyak)
set(XBYAK_CLONE_DIR     ${XBYAK_PREFIX_DIR}/src/${XBYAK_PROJECT})
set(XBYAK_INSTALL_ROOT  ${ANAKIN_THIRD_PARTY_PATH}/xbyak)
set(XBYAK_INC_DIR       ${XBYAK_INSTALL_ROOT}/include)

message(STATUS "Scanning external modules ${Green}xbyak${ColourReset} ...")


include_directories(${XBYAK_INC_DIR})

if(USE_SGX)
    set(SGX_PATCH_CMD "cd ${ANAKIN_THIRD_PARTY_PATH} && patch -p0 <xbyak.patch")
else()
    # use a whitespace as nop so that sh won't complain about missing argument
    set(SGX_PATCH_CMD " ")
endif()

ExternalProject_Add(
    ${XBYAK_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ""
    GIT_REPOSITORY      "https://github.com/herumi/xbyak.git"
    GIT_TAG             "v5.661"  # Jul 26th
    PREFIX              ${XBYAK_PREFIX_DIR}/src
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${XBYAK_INSTALL_ROOT}
    INSTALL_COMMAND     make install
    COMMAND             sh -c "${SGX_PATCH_CMD}"
    VERBATIM
)

add_library(xbyak SHARED IMPORTED GLOBAL)
add_dependencies(xbyak ${XBYAK_PROJECT})

list(APPEND ANAKIN_SABER_DEPENDENCIES xbyak)
