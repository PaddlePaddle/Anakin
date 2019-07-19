# Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

#!/bin/bash
# This script shows how one can build a anakin for the <NVIDIA> gpu platform 
ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

echo "delete cache files"
rm -rf ~/.cache/amd_saber/

# build the target into gpu_build.
BUILD_ROOT=$ANAKIN_ROOT/amd_gpu_build

export OCL_ROOT=/opt/rocm/opencl

mkdir -p $BUILD_ROOT
echo "-- Build anakin gpu(AMD) into: $BUILD_ROOT"

# Now, actually build the gpu target.
echo "-- Building anakin ..."
cd $BUILD_ROOT

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ARM_PLACE=NO \
    -DUSE_GPU_PLACE=YES \
    -DUSE_NV_GPU=NO \
    -DAMD_GPU=YES \
    -DNVIDIA_GPU=NO \
    -DUSE_X86_PLACE=NO \
    -DUSE_BM_PLACE=NO \
    -DBUILD_WITH_UNIT_TEST=YES \
    -DUSE_PYTHON=OFF \
    -DENABLE_DEBUG=NO \
    -DENABLE_VERBOSE_MSG=NO \
    -DENABLE_AMD_PROFILING=NO \
    -DENABLE_AMD_DO_SEARCH=NO \
    -DENABLE_AMD_EXPAND_ALL_SEARCH=NO \
    -DDISABLE_ALL_WARNINGS=YES \
    -DENABLE_NOISY_WARNINGS=NO \
    -DUSE_OPENMP=NO\
    -DBUILD_SHARED=YES\
    -DUSE_OPENCL=YES \
    -DUSE_OPENCV=NO \
    -DBUILD_EXAMPLES=NO \
    -DBUILD_WITH_LITE=NO \
    ..
        

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)"
fi

