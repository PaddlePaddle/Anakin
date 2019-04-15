#!/bin/bash
# This script shows how one can build a anakin for the <NVIDIA> gpu platform 
set -e

ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

# build the target into gpu_build.
BUILD_ROOT=$ANAKIN_ROOT/lite_build

mkdir -p $BUILD_ROOT
echo "-- Build anakin lite into: $BUILD_ROOT"
export PATH=/Users/scmtools/buildkit/cmake/cmake-3.8.2/bin:$PATH
# Now, actually build the gpu target.
echo "-- Building anakin ..."
cd $BUILD_ROOT

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
	-DUSE_ARM_PLACE=NO \
	-DUSE_GPU_PLACE=NO \
	-DUSE_X86_PLACE=NO \
	-DBUILD_WITH_UNIT_TEST=NO \
   	-DUSE_PYTHON=OFF \
	-DENABLE_DEBUG=NO \
	-DENABLE_VERBOSE_MSG=NO \
        -DENABLE_MIN_DEPENDENCY=YES \
        -DPROTOBUF_ROOT=/Users/scmbuild/workspaces_cluster/baidu.sys-hic-gpu.Anakin-2.0/baidu/sys-hic-gpu/Anakin-2.0/protobuf/ \
	-DDISABLE_ALL_WARNINGS=YES \
	-DENABLE_NOISY_WARNINGS=NO \
    -DUSE_OPENMP=NO \
	-DBUILD_SHARED=YES \
	-DBUILD_EXAMPLES=NO \
	-DBUILD_LITE=YES

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" install
else
    make "-j$(nproc)" install   
fi

