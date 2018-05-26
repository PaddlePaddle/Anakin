# ----------------------------------------------------------------------------
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
# @file     CMakeLists files in the saber  subdirectory for nvidia gpu code
# @auther   cuichaowen
# @date     2017-11-29
# ----------------------------------------------------------------------------

if(USE_CUDA)
    anakin_fetch_files_with_suffix(${CUDA_BASE_CODE_ROOT}/cuda_c "cu" ANAKIN_SABER_CUDA_C_SRC)
    anakin_fetch_files_with_suffix(${CUDA_BASE_CODE_ROOT}/sass/sm_${SELECTED_SASS_TARGET_ARCH} "a" ANAKIN_SABER_SASS_STATIC_LIB)
endif()

macro(anakin_set_upscope src)
    set(${src} ${${src}} PARENT_SCOPE)
endmacro()

if(USE_CUDA)
    anakin_set_upscope(ANAKIN_SABER_CUDA_C_SRC)
    anakin_set_upscope(ANAKIN_SABER_SASS_STATIC_LIB)
endif()
