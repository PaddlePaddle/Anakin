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
set(CMAKE_CXX_STANDARD_REQUIRED ON)
anakin_add_compile_option(-std=c++11)
anakin_add_compile_option(-fPIC)

if(USE_X86_PLACE)
    anakin_add_compile_option(-march=native)
    if (BUILD_X86_TARGET MATCHES "knl" OR ${BUILD_X86_ARCH} MATCHES "knl")
        anakin_add_compile_option(-mavx512bw)
        anakin_add_compile_option(-mavx512f)
    endif ()
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
anakin_add_compile_option(-Wno-missing-field-initializers)
anakin_add_compile_option(-Wno-extra)

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

if(USE_SGX)
    # SGX build uses MKL instead of MKLMKL, possibly a higer version
    # Some APIs may be deprecated by later MKL. We want to ignore
    # these warnings
    anakin_add_compile_option(-Wno-deprecated-declarations)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    anakin_add_compile_option(-O0)
    anakin_add_compile_option(-g)
    anakin_add_compile_option(-gdwarf-2) # for old version gcc and gdb. see: http://stackoverflow.com/a/15051109/673852
else()
    if(USE_SGX)
      anakin_add_compile_option(-Os)
    else()
      anakin_add_compile_option(-Ofast)
    endif()

    if(USE_ARM_PLACE)
        add_compile_options(-Ofast)
        add_compile_options(-ffast-math)
        add_compile_options(-Os)
    endif()

    anakin_add_compile_option(-DNDEBUG)
endif()

if(TARGET_ANDROID)
    anakin_add_compile_option(-pie)
    add_compile_options(-ldl)
    anakin_add_compile_option(-lc)
    set(ANAKIN_EXTRA_CXX_FLAGS "${ANAKIN_EXTRA_CXX_FLAGS} ${ANDROID_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
    set(MAKE_STATIC_LINKER_FLAGS "${MAKE_STATIC_LINKER_FLAGS} -Wl,--gc-sections")
endif()

if(TARGET_IOS)
    # none temp
endif()

if(BUILD_STATIC OR X86_COMPILE_482)
    anakin_add_compile_option(-static-libstdc++)
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
    anakin_add_compile_option(-D_GLIBCXX_USE_CXX11_ABI=0) #use std namespace for string and list rather than std::__CXX11::
#    anakin_add_compile_option(-static-libstdc++)
#    anakin_add_compile_option(-static-libgcc)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    anakin_add_compile_option(-fabi-version=6)
    anakin_add_compile_option(-fabi-compat-version=2) #add compat
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

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ANAKIN_EXTRA_CXX_FLAGS}")

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
endif()
