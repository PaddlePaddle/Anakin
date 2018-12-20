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

# ----------------------------------------------------------------------------
# section: set the compiler and linker options 
# ----------------------------------------------------------------------------
set(ANAKIN_EXTRA_CXX_FLAGS "")
set(ANAKIN_NVCC_FLAG "")
set(CMAKE_CXX_STANDARD 11)

anakin_add_compile_option(-fPIC)

if(NOT USE_SGX)
    anakin_add_compile_option(-ldl)
    anakin_add_compile_option(-pthread)
    if(USE_ARM_PLACE)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    else()
        anakin_add_compile_option(-lrt)
    endif()
endif()

anakin_add_compile_option(-W)
anakin_add_compile_option(-Wall)
anakin_add_compile_option(-Werror=return-type)
anakin_add_compile_option(-Werror=address)
anakin_add_compile_option(-Werror=sequence-point)
anakin_add_compile_option(-Wno-unused-variable) # no unused-variable
anakin_add_compile_option(-Wformat)
anakin_add_compile_option(-Wmissing-declarations)
anakin_add_compile_option(-Winit-self)
anakin_add_compile_option(-Wpointer-arith)
anakin_add_compile_option(-Wshadow)
anakin_add_compile_option(-fpermissive)
anakin_add_compile_option(-Wsign-promo)
anakin_add_compile_option(-fdiagnostics-show-option)

if(ENABLE_NOISY_WARNINGS)
	anakin_add_compile_option(-Wcast-align)
	anakin_add_compile_option(-Wstrict-aliasing=2)
	anakin_add_compile_option(-Wundef)
	anakin_add_compile_option(-Wsign-compare)
else()
	anakin_add_compile_option(-Wno-undef)
	anakin_add_compile_option(-Wno-narrowing)
	anakin_add_compile_option(-Wno-unknown-pragmas)
	anakin_add_compile_option(-Wno-delete-non-virtual-dtor)
	anakin_add_compile_option(-Wno-comment)
	anakin_add_compile_option(-Wno-sign-compare)
    anakin_add_compile_option(-Wno-write-strings) 
    anakin_add_compile_option(-Wno-ignored-qualifiers) 
    anakin_add_compile_option(-Wno-enum-compare)
    anakin_add_compile_option(-Wno-missing-field-initializers)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    anakin_add_compile_option(-O0)
	anakin_add_compile_option(-g)
	anakin_add_compile_option(-gdwarf-2) # for old version gcc and gdb. see: http://stackoverflow.com/a/15051109/673852 
else()
    if(NOT USE_SGX)
    anakin_add_compile_option(-O3)
    endif()
	anakin_add_compile_option(-Ofast)
	anakin_add_compile_option(-DNDEBUG)
endif()

if(TARGET_ANDROID)
	anakin_add_compile_option(-pie)
	anakin_add_compile_option(-mfloat-abi=softfp)
	anakin_add_compile_option(-mfpu=neon)
	anakin_add_compile_option(-ffast-math)
	anakin_add_compile_option(-lc)
	set(ANAKIN_EXTRA_CXX_FLAGS "${ANAKIN_EXTRA_CXX_FLAGS} ${ANDROID_CXX_FLAGS}")
endif()

if(TARGET_IOS)
	# none temp
endif()

if(USE_X86_PLACE)
if(X86_COMPILE_482)
    set(CMAKE_SYSROOT /opt/compiler/gcc-4.8.2/)
    set(CMAKE_SKIP_BUILD_RPATH TRUE)
    set(CMAKE_BUILD_RPATH "/opt/compiler/gcc-4.8.2/lib64/")
    set(CMAKE_INSTALL_RPATH "/opt/compiler/gcc-4.8.2/lib64/;${PROJECT_SOURCE_DIR}/${AK_OUTPUT_PATH}/;./;../")
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,-dynamic-linker,/opt/compiler/gcc-4.8.2/lib64/ld-linux-x86-64.so.2")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    anakin_add_compile_option(-static-libstdc++)
#    anakin_add_compile_option(-static-libgcc)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	anakin_add_compile_option(-fabi-version=6)
    anakin_add_compile_option(-march=${BUILD_X86_ARCH})
endif()
if(USE_OPENMP)
    anakin_add_compile_option(-fopenmp)
endif()
    anakin_add_compile_option(-Wall)
    anakin_add_compile_option(-Wno-comment)
    anakin_add_compile_option(-Wno-unused-local-typedefs)
endif()

# The -Wno-long-long is required in 64bit systems when including sytem headers.
if(X86_64)
	anakin_add_compile_option(-Wno-long-long)
endif()

set(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} ${ANAKIN_EXTRA_CXX_FLAGS})

#if(WIN32) 
#    if(MSVC)
#    	message(STATUS "Using msvc compiler")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SCL_SECURE_NO_WARNINGS")
#    endif()
#endif()

if(USE_CUDA)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        anakin_add_compile_option("-Xcompiler -fPIC" NVCC)
        anakin_add_compile_option(-G NVCC)
        anakin_add_compile_option(-g NVCC)
        anakin_add_compile_option(-std=c++11 NVCC)
        anakin_add_compile_option("--default-stream per-thread" NVCC)
        anakin_add_compile_option(-Wno-deprecated-gpu-targets NVCC) # suppress warning by architectures are deprecated (2.0,2.1)
    else()
        anakin_add_compile_option("-Xcompiler -fPIC" NVCC)
        anakin_add_compile_option(-O3 NVCC)
        anakin_add_compile_option(-std=c++11 NVCC)
        anakin_add_compile_option("--default-stream per-thread" NVCC)
        anakin_add_compile_option(-Wno-deprecated-gpu-targets NVCC)
    endif()
    # set default nvidia gpu arch
    set(ANAKIN_ARCH_LIST "3.5;5.0;6.0;6.1")
endif()
