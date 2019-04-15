# ----------------------------------------------------------------------------
# section: Find mlu and config compile options.
# ----------------------------------------------------------------------------
macro(anakin_find_mlulib)
  SET(CNRTML_ROOT ${ANAKIN_THIRD_PARTY_PATH}/mlu)
  SET(CNML_INCLUDE_SEARCH_PATHS ${CNRTML_ROOT}/include)
  SET(CNML_LIB_SEARCH_PATHS ${CNRTML_ROOT}/lib)

  SET(CNRT_INCLUDE_SEARCH_PATHS ${CNRTML_ROOT}/include)
  SET(CNRT_LIB_SEARCH_PATHS ${CNRTML_ROOT}/lib)

  find_path(CNML_INCLUDE_DIR cnml.h PATHS ${CNML_INCLUDE_SEARCH_PATHS} NO_DEFAULT_PATH)

  find_path(CNRT_INCLUDE_DIR cnrt.h PATHS ${CNRT_INCLUDE_SEARCH_PATHS} NO_DEFAULT_PATH)

  find_library(CNML_LIBRARY NAMES libcnml.so
               PATHS ${CNML_LIB_SEARCH_PATHS}
               DOC "library path for cnml.")

  find_library(CNRT_LIBRARY NAMES libcnrt.so
               PATHS ${CNRT_LIB_SEARCH_PATHS}
               DOC "library path for cnrt.")

	if(CNML_INCLUDE_DIR AND CNML_LIBRARY AND CNRT_INCLUDE_DIR AND CNRT_LIBRARY)
		set(MLU_FOUND YES)
	endif()
	if(MLU_FOUND)
		include_directories(SYSTEM ${CNML_INCLUDE_DIR})
		list(APPEND ANAKIN_LINKER_LIBS ${CNML_LIBRARY})
    message(STATUS "Found CNML (include: ${CNML_INCLUDE_DIR}, library: ${CNML_LIBRARY})")

		include_directories(SYSTEM ${CNRT_INCLUDE_DIR})
		list(APPEND ANAKIN_LINKER_LIBS ${CNRT_LIBRARY})
    message(STATUS "Found CNRT (include: ${CNRT_INCLUDE_DIR}, library: ${CNRT_LIBRARY})")

	else()
#		message(SEND_ERROR "Could not find cnml library in: ${CNML_ROOT}")
#		message(SEND_ERROR "Could not find cnrt library in: ${CNRT_ROOT}")
		message(STATUS "Could not find cnml library in: ${CNML_ROOT}")
		message(STATUS "Could not find cnrt library in: ${CNRT_ROOT}")
	endif()
endmacro()
