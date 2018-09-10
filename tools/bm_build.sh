#!/bin/bash
# This script shows how one can build a anakin for the <BitMain> TPU platform
ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

# build the target into gpu_build.
BUILD_ROOT=$ANAKIN_ROOT/bm_build

mkdir -p $BUILD_ROOT
echo "-- Build anakin TPU(BitMain) into: $BUILD_ROOT"

# Now, actually build the gpu target.
echo "-- Building anakin ..."
cd $BUILD_ROOT


cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
	-DUSE_ARM_PLACE=NO \
	-DUSE_GPU_PLACE=NO \
	-DUSE_NV_GPU=NO \
	-DAMD_GPU_GPU=NO \
	-DUSE_X86_PLACE=NO \
	-DUSE_BM_PLACE=YES \
	-DBUILD_FAT_BIN=YES \
	-DBUILD_WITH_UNIT_TEST=YES \
   	-DUSE_PYTHON=OFF \
	-DENABLE_DEBUG=NO \
	-DENABLE_VERBOSE_MSG=NO \
	-DDISABLE_ALL_WARNINGS=YES \
	-DENABLE_NOISY_WARNINGS=NO \
    -DUSE_OPENMP=YES\
	-DBUILD_SHARED=YES\
	-DBUILD_WITH_FRAMEWORK=NO

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)"   && make install
fi

