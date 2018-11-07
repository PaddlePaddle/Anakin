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

if(USE_CUDA)
    anakin_fetch_files_with_suffix(${CUDA_BASE_CODE_ROOT}/cuda_c "cu" ANAKIN_SABER_CUDA_C_SRC)
    set(SASS_LIB_PATH ${ANAKIN_THIRD_PARTY_PATH}/sass/lib/sm_${SELECTED_SASS_TARGET_ARCH}/)
    list(APPEND ANAKIN_SABER_SASS_STATIC_LIB ${SASS_LIB_PATH}/libanakin_saber_sass_sm_${SELECTED_SASS_TARGET_ARCH}.a)
endif()

macro(anakin_set_upscope src)
    set(${src} ${${src}} PARENT_SCOPE)
endmacro()

if(USE_CUDA)
    anakin_set_upscope(ANAKIN_SABER_CUDA_C_SRC)
    anakin_set_upscope(ANAKIN_SABER_SASS_STATIC_LIB)
endif()
