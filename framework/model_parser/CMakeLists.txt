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
set(ANAKIN_BASE_SRC "")

# add ak_base_source files
anakin_fetch_files_with_suffix(${ANAKIN_MODEL_PARSER}/parser "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_MODEL_PARSER}/proto "cpp" ANAKIN_BASE_SRC)

list(APPEND ANAKIN_SRC ${ANAKIN_BASE_SRC})
set(ANAKIN_SRC ${ANAKIN_SRC} PARENT_SCOPE)
unset(ANAKIN_BASE_SRC)
