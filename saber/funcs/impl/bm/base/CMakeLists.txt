if(USE_BM)
    anakin_fetch_files_with_suffix_recursively(${BM_BASE_CODE_ROOT}/include "h" ANAKIN_SABER_BM_C_SRC)
    anakin_fetch_files_with_suffix_recursively(${BM_BASE_CODE_ROOT}/lib "o" ANAKIN_SABER_BM_STATIC_LIB)
endif()

macro(anakin_set_upscope src)
    set(${src} ${${src}} PARENT_SCOPE)
endmacro()

if(USE_BM)
    anakin_set_upscope(ANAKIN_SABER_BM_C_SRC)
    anakin_set_upscope(ANAKIN_SABER_BM_STATIC_LIB)
endif()
