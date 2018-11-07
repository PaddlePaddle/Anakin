#!/bin/bash
set -e
# This script shows how one can build a anakin for the <NVIDIA> gpu platform
ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

# build the target into gpu_build.
BUILD_ROOT=$ANAKIN_ROOT/nv_gpu_build

mkdir -p $BUILD_ROOT
echo "-- Build anakin gpu(NVIDIA) into: $BUILD_ROOT"

# Now, actually build the gpu target.
echo "-- Building anakin ..."
cd $BUILD_ROOT


cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
	-DUSE_ARM_PLACE=NO \
	-DUSE_GPU_PLACE=YES \
    -DNVIDIA_GPU=YES \
    -DAMD_GPU=NO \
	-DUSE_X86_PLACE=YES \
	-DUSE_BM_PLACE=NO \
	-DBUILD_FAT_BIN=YES \
	-DBUILD_WITH_UNIT_TEST=YES \
    -DBUILD_RPC=OFF \
   	-DUSE_PYTHON=OFF \
    -DUSE_GFLAGS=OFF \
	-DENABLE_DEBUG=OFF \
	-DENABLE_VERBOSE_MSG=NO \
	-DDISABLE_ALL_WARNINGS=YES \
	-DENABLE_NOISY_WARNINGS=NO \
    -DUSE_OPENMP=YES\
	-DBUILD_SHARED=YES\
	-DBUILD_WITH_FRAMEWORK=YES

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)"   && make install
fi

