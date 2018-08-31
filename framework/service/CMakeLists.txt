# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# ----------------------------------------------------------------------------

# used for temporary
anakin_fetch_include_recursively(${ANAKIN_SERVICE})
set(ANAKIN_BASE_SRC "")

anakin_fetch_files_with_suffix(${ANAKIN_SERVICE}/api "cpp" ANAKIN_BASE_SRC)
anakin_fetch_files_with_suffix(${ANAKIN_SERVICE} "cpp" ANAKIN_BASE_SRC)

list(APPEND ANAKIN_SRC ${ANAKIN_BASE_SRC})
set(ANAKIN_SRC ${ANAKIN_SRC} PARENT_SCOPE)
unset(ANAKIN_BASE_SRC)
