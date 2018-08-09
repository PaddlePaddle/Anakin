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

#anakin cmake module
set(CMAKE_MODULE_PATH "${ANAKIN_ROOT}/cmake")

set(ANAKIN_LINKER_LIBS "")

if(UNIX)
	if(USE_ARM_PLACE )
	elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	else()
		find_library(RTLIB rt)
		if(RTLIB)
			list(APPEND ANAKIN_LINKER_LIBS ${RTLIB})
		else()
			message(SEND_ERROR "Could not found -lrt !")
		endif()
	endif()

    find_library(DLLIB dl)
    if(DLLIB)
        list(APPEND ANAKIN_LINKER_LIBS ${DLLIB})
    else()
        message(SEND_ERROR "Could not found -ldl !")
    endif()
endif()

#find opencv version >= 2.4.3
macro(anakin_find_opencv)

	if(USE_ARM_PLACE AND TARGET_ANDROID)
		include_directories(${CMAKE_SOURCE_DIR}/third-party/arm-android/opencv/sdk/native/jni/include/)
		LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third-party/arm-android/opencv/sdk/native/libs/armeabi-v7a/)

	else()

		if(BUILD_SHARED) # temporary not support static link opencv.
			find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
			if(NOT OpenCV_FOUND)
				find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
			endif()
			if(OpenCV_FOUND)
				message(STATUS "Found opencv: ${OpenCV_INCLUDE_DIRS}")
				include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
				list(APPEND ANAKIN_LINKER_LIBS ${OpenCV_LIBS})

			else()
				message(SEND_ERROR "Could not found opencv !")
			endif()
		else() # BUILD_STATIC
			set(OPENCV_LIB_PATH "" CACHE "Path to oopen cv library")
			list(APPEND OPENCV_STATIC_LIBS ${OPENCV_LIB_PATH}/libopencv_core.a
					${OPENCV_LIB_PATH}libopencv_highgui.a
					${OPENCV_LIB_PATH}libopencv_imgproc.a
					${OPENCV_LIB_PATH}libopencv_contrib.a)
			foreach(CV_LIB ${OPENCV_STATIC_LIBS})
				list(APPEND ANAKIN_LINKER_LIBS ${CV_LIB})
			endforeach()
			unset(__CV_LIB_FULL_PATH)
		endif()

    endif()
endmacro()

#find opencl 
macro(anakin_find_opencl)
	set(OCL_ROOT "" CACHE PATH "openCL root dir.")

	find_path(OCL_INCLUDE_DIR  NAMES  CL/cl.h PATHS ${OCL_ROOT}/include $ENV{OCL_ROOT}/include)

	find_library(OCL_LIBRARIES NAMES libOpenCL.so PATHS ${OCL_ROOT})
	if(OCL_INCLUDE_DIR AND OCL_LIBRARIES)
		set(OCL_FOUND  TRUE)
		message(STATUS "Found opencl: ${OCL_INCLUDE_DIR}")
                include_directories(SYSTEM ${OCL_INCLUDE_DIR})
                list(APPEND ANAKIN_LINKER_LIBS ${OCL_LIBRARIES})
	else()
		message(SEND_ERROR "Could not found opencl !")
	endif()
endmacro()


#find boost version >= 1.59.0
macro(anakin_find_boost)
	#when using some older versions of boost with cmake-2.8.6-rc2 or later
	ADD_DEFINITIONS(-DBoost_NO_BOOST_CMAKE)

	set(Boost_USE_STATIC_LIBS        ON) # only find static libs
	set(Boost_USE_MULTITHREADED      ON)
	set(Boost_USE_STATIC_RUNTIME     ON)
	set(BOOST_ROOT ${ANAKIN_ROOT}/thirdparty/boost/)
	find_package(Boost 1.59.0 QUIET COMPONENTS thread variant)
	if(Boost_FOUND)
		include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
		list(APPEND ANAKIN_LINKER_LIBS ${Boost_LIBRARIES})	
	endif()	
endmacro()

#find intel mkl lib.
macro(anakin_find_mkl)
	set(INTEL_ROOT "/opt/intel" CACHE PATH "Folder contains intel libs.")
	set(MKL_ROOT "" CACHE PATH "Folder contains intel(R) mkl libs.")	
	# options for mkl
	set(MKL_USE_SINGLE_DYNAMIC_LIBRARY YES)
	set(MKL_USE_STATIC_LIBS NO)
	if(NOT BUILD_SHARED)
		set(MKL_USE_STATIC_LIBS YES)
	endif()
	anakin_option(MKL_MULTI_THREADED  "Use multi-threading"   ON IF NOT MKL_USE_SINGLE_DYNAMIC_LIBRARY)

	find_path(MKL_INCLUDE_DIR mkl_blas.h PATHS $ENV{MKL_ROOT}/include
    										   ${INTEL_ROOT}/mklml/include
    										   DOC "Folder contains MKL")
	#message(STATUS "test ??? ${MKL_INCLUDE_DIR}")
	# include to anakin sys.
	include_directories(SYSTEM ${MKL_INCLUDE_DIR})
	# find for libs.
	if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    	set(__path_suffixes lib lib/ia32)
  	else()
    	set(__path_suffixes lib lib/intel64)
  	endif()
	set(__mkl_libs "")
	if(MKL_USE_SINGLE_DYNAMIC_LIBRARY)
  		list(APPEND __mkl_libs rt)
	else()	
		if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    		if(WIN32)
      			list(APPEND __mkl_libs intel_c)
    		else()
      			list(APPEND __mkl_libs intel gf)
    		endif()
  		else()
    		list(APPEND __mkl_libs intel_lp64 gf_lp64)
  		endif()		

		if(MKL_MULTI_THREADED)
    		list(APPEND __mkl_libs intel_thread)
  		else()
     		list(APPEND __mkl_libs sequential)
  		endif()

		list(APPEND __mkl_libs core cdft_core)
	endif()
	set(__trigger_mkllib FALSE)
	foreach (__lib ${__mkl_libs})
		set(__mkl_lib "mkl_${__lib}")
    	string(TOUPPER ${__mkl_lib} __mkl_lib_upper)
		if(MKL_USE_STATIC_LIBS)
      		set(__mkl_lib "lib${__mkl_lib}.a")
    	endif()
		find_library(${__mkl_lib_upper}_LIBRARY NAMES ${__mkl_lib}
      											PATHS ${MKL_ROOT} ${MKL_INCLUDE_DIR}/..
      											PATH_SUFFIXES ${__path_suffixes}
      											DOC "The path to Intel(R) MKL ${__mkl_lib} library")
		if(${__mkl_lib_upper}_LIBRARY)
    		list(APPEND ANAKIN_LINKER_LIBS ${${__mkl_lib_upper}_LIBRARY})
			#message(STATUS " -----------  ${${__mkl_lib_upper}_LIBRARY}")
			set(__trigger_mkllib TRUE)
		endif()
	endforeach()
	
	if(NOT MKL_USE_SINGLE_DYNAMIC_LIBRARY)
  		if (MKL_USE_STATIC_LIBS)
    		set(__iomp5_libs iomp5 libiomp5mt.lib)
  		else()
    		set(__iomp5_libs iomp5 libiomp5md.lib)
  		endif()

  		find_library(MKL_RTL_LIBRARY NAMES ${__iomp5_libs}
     								 PATHS ${INTEL_ROOT}/compiler ${MKL_ROOT} ${MKL_INCLUDE_DIR}/.. ${MKL_ROOT}/../compiler
     								 PATH_SUFFIXES ${__path_suffixes}
     								 DOC "Path to Path to OpenMP runtime library")

		list(APPEND ANAKIN_LINKER_LIBS ${MKL_RTL_LIBRARY})
		#message(STATUS " +++++++++++  ${MKL_RTL_LIBRARY}")
	endif()

	if((MKL_INCLUDE_DIR) AND (__trigger_mkllib OR MKL_RTL_LIBRARY))
		set(MKL_FOUND YES)
    endif()

	if(MKL_FOUND)
  		message(STATUS "Found Intel(R) MKL: ${MKL_INCLUDE_DIR} .")
	else()
		message(FATAL_ERROR "Could not found mkl !")
	endif()
	
endmacro()

# find glog and config it
macro(anakin_find_glog)
	set(CMAKE_PREFIX_PATH ${ANAKIN_ROOT}/thirdparty/glog/build)
	find_path(GLOG_INCLUDE_DIR raw_logging.h ${ANAKIN_ROOT}/thirdparty/glog/build/glog)
	find_library(GLOG_LIBRARY NAMES libglog.a
                                   PATHS ${GLOG_INCLUDE_DIR}/../
                                   DOC "library path for glog.")
	if(GLOG_INCLUDE_DIR AND GLOG_LIBRARY)
		set(GLOG_FOUND TRUE)
	endif()
	if(GLOG_FOUND)
    	message(STATUS "Found glog in ${GLOG_INCLUDE_DIR}")
		include_directories(SYSTEM ${GLOG_INCLUDE_DIR})
      		list(APPEND ANAKIN_LINKER_LIBS ${GLOG_LIBRARY})
  	endif()
endmacro()

# find gtest
macro(anakin_find_gtest)
	find_path(GTEST_INCLUDE_DIR gtest.h ${ANAKIN_ROOT}/thirdparty/googletest/include/gtest)
    find_library(GTEST_LIBRARY NAMES libgtest.a
                                   PATHS ${GTEST_INCLUDE_DIR}/../../
                                   DOC "library path for gtest.")

    if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY)
    	set(GTEST_FOUND TRUE)
	endif()
    if(GTEST_FOUND)
        message(STATUS "Found gtest in ${GTEST_INCLUDE_DIR}")
		include_directories(${GTEST_INCLUDE_DIR}/../)
        #list(APPEND ANAKIN_LINKER_LIBS ${GTEST_LIBRARY})
		list(APPEND ANAKIN_LINKER_LIBS ${GTEST_LIBRARY})
    endif()
endmacro()


macro(anakin_find_gflags)
	set(GFLAGS_INCLUDE_DIR ${ANAKIN_ROOT}/third-party/gflags/include)
    find_library(GFLAGS_LIBRARY NAMES libgflags.so
                                   PATHS ${GFLAGS_INCLUDE_DIR}/../lib
                                   DOC "library path for gflags.")
    if(GFLAGS_INCLUDE_DIR AND GFLAGS_LIBRARY)
    	set(GFLAGS_FOUND TRUE)
    endif()
    if(GFLAGS_FOUND)
    	message(STATUS "Found gflags in ${GFLAGS_INCLUDE_DIR}")
    	include_directories(${GFLAGS_INCLUDE_DIR})
    	list(APPEND ANAKIN_LINKER_LIBS ${GFLAGS_LIBRARY})
    endif()
endmacro()

macro(anakin_find_xbyak)
        set(XBYAK_ROOT ${ANAKIN_ROOT}/third-party/xbyak)
        find_path(XBYAK_ROOT_INCLUDE xbyak.h ${XBYAK_ROOT}/include/xbyak)
        if (XBYAK_ROOT_INCLUDE)
            set(XBYAK_FOUND TRUE)
        endif()
        if (XBYAK_FOUND)
            message(STATUS "FOUND XBYAK in ${XBYAK_ROOT}")
            include_directories(${XBYAK_ROOT}/include)
        else()
            message(FATAL_ERROR "NOT FOUND XBYAK")
        endif()
endmacro()

macro(anakin_find_mklml)
        set(MKLML_ROOT ${ANAKIN_ROOT}/third-party/mklml)
        find_path(MKLML_ROOT_INCLUDE mkl_vsl.h ${MKLML_ROOT}/include)
        if (MKLML_ROOT_INCLUDE)
            set(MKLML_FOUND TRUE)
        endif()
        if (MKLML_FOUND)
            message(STATUS "FOUND MKLML in ${MKLML_ROOT}")
            include_directories(${MKLML_ROOT}/include)
            set(MKLML_LIBRARIES "")
            list(APPEND MKLML_LIBRARIES ${MKLML_ROOT}/lib/libiomp5.so)
            list(APPEND MKLML_LIBRARIES ${MKLML_ROOT}/lib/libmklml_intel.so)
            list(APPEND ANAKIN_LINKER_LIBS ${MKLML_LIBRARIES})
        else()
                message(FATAL_ERROR "NOT FOUND MKLML")
        endif()
endmacro()

macro(anakin_find_protobuf)
	if(USE_ARM_PLACE)
		set(ARM_RPOTO_ROOT "${CMAKE_SOURCE_DIR}/third-party/arm-android/protobuf")
		include_directories(${ARM_RPOTO_ROOT}/include)
		set(PROTOBUF_LIBRARIES "")
		#if(BUILD_SHARED)
		#	list(APPEND ANAKIN_LINKER_LIBS ${ARM_RPOTO_ROOT}/lib/libprotobuf.so)
		#else()
			list(APPEND ANAKIN_LINKER_LIBS ${ARM_RPOTO_ROOT}/lib/libprotobuf.a)
		#endif()
		find_library( # Sets the name of the path variable.
				log-lib

				# Specifies the name of the NDK library that
				# you want CMake to locate.
				log )
		list(APPEND ANAKIN_LINKER_LIBS ${log-lib})
	else()
		find_package(Protobuf REQUIRED)
		if(PROTOBUF_FOUND)
			message(STATUS "Found protobuf in ${PROTOBUF_INCLUDE_DIR}")
			include_directories(${PROTOBUF_INCLUDE_DIR})
			list(APPEND ANAKIN_LINKER_LIBS ${PROTOBUF_LIBRARIES})
		endif()
	endif()
endmacro()


macro(anakin_find_openmp)
	find_package(OpenMP REQUIRED)
	if(OPENMP_FOUND OR OpenMP_CXX_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		message(STATUS "Found openmp in ${OPENMP_INCLUDE_DIR}")
		message(STATUS " |--openmp cflags: ${OpenMP_C_FLAGS}")
		message(STATUS " |--openmp cxxflags: ${OpenMP_CXX_FLAGS}")
		message(STATUS " |--openmp cflags: ${OpenMP_EXE_LINKER_FLAGS}")
	else()
		message(FATAL_ERROR "Could not found openmp !")
	endif()
endmacro()
