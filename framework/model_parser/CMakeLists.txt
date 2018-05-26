# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# @file     CMakeLists files in the model parser directory of project
# @auther   cuichaowen
# @date     2017-10-24
# ----------------------------------------------------------------------------

set(ANAKIN_BASE_SRC "")

# add ak_base_source files
anakin_fetch_files_with_suffix(${ANAKIN_MODEL_PARSER}/parser "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_MODEL_PARSER}/proto "cpp" ANAKIN_BASE_SRC)

list(APPEND ANAKIN_SRC ${ANAKIN_BASE_SRC})
set(ANAKIN_SRC ${ANAKIN_SRC} PARENT_SCOPE)
unset(ANAKIN_BASE_SRC)
