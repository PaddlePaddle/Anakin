#!/bin/bash
# This script shows how one can build a anakin for the <X86> platform
ANAKIN_ROOT="$( cd "$(dirname "$0")"/../.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

# build the target into gpu_build.
BUILD_ROOT=$ANAKIN_ROOT/x86_build_$1
BUILD_OUT_NAME=x86_$1
BUILD_OUT_PATH=$ANAKIN_ROOT/$BUILD_OUT_NAME

rm -fr $BUILD_ROOT
rm -fr $BUILD_OUT_PATH
mkdir -p $BUILD_ROOT
mkdir -p $BUILD_OUT_PATH
echo "-- Build anakin x86 into: $BUILD_ROOT"

# Now, actually build the gpu target.
echo "-- Building anakin ..."
cd $BUILD_ROOT

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
	-DUSE_ARM_PLACE=NO \
	-DUSE_GPU_PLACE=NO \
	-DUSE_NV_GPU=NO \
	-DAMD_GPU_GPU=NO \
	-DUSE_X86_PLACE=YES \
	-DUSE_BM_PLACE=NO \
	-DBUILD_WITH_UNIT_TEST=YES \
   	-DUSE_PYTHON=OFF \
	-DENABLE_DEBUG=NO \
	-DENABLE_VERBOSE_MSG=NO \
	-DDISABLE_ALL_WARNINGS=YES \
	-DENABLE_NOISY_WARNINGS=NO \
    -DUSE_OPENMP=YES\
	-DBUILD_SHARED=YES\
	-DBUILD_WITH_FRAMEWORK=NO\
	-DBUILD_X86_TARGET=$1\
	-DAK_OUTPUT_PATH=$BUILD_OUT_NAME

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)"
else
    make "-j$(nproc)"
fi

