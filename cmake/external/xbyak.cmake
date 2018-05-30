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
set(XBYAK_INSTALL_ROOT  ${ANAKIN_THIRD_PARTY_PATH}/xbyak)
set(XBYAK_INC_DIR       ${XBYAK_INSTALL_ROOT}/include)

message(STATUS "Scanning external modules ${Green}xbyak${ColourReset} ...")


include_directories(${XBYAK_INC_DIR})
include_directories(${XBYAK_INC_DIR}/xbyak)

ExternalProject_Add(
    ${XBYAK_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ""
    GIT_REPOSITORY      "https://github.com/herumi/xbyak.git"
    GIT_TAG             "fe083912c8ac7b7e2b0081cbd6213997bc8b56e6"  # mar 6, 2018
    PREFIX              ${XBYAK_PREFIX_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${XBYAK_INSTALL_ROOT}
)

add_library(xbyak INTERFACE)
add_dependencies(xbyak ${XBYAK_PROJECT})
list(APPEND ANAKIN_SABER_DEPENDENCIES xbyak)
