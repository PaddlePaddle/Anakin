#!/bin/bash
set -e
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
	-DBUILD_WITH_FRAMEWORK=YES\
	-DBUILD_X86_TARGET=$1\
	-DENABLE_MIN_DEPENDENCY=YES \
	-DAK_OUTPUT_PATH=$BUILD_OUT_NAME\
	-DX86_COMPILE_482=YES

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)" && make install
fi

if ! [ -e "$ANAKIN_ROOT/x86_output" ]; then
    mkdir -p "$ANAKIN_ROOT/x86_output"
fi
SO_PATH=$ANAKIN_ROOT/x86_output/$1
mkdir -p $SO_PATH

if ! [ -e "$ANAKIN_ROOT/x86_output/anakin_runner.h" ]; then
    cp -r $BUILD_OUT_PATH/framework/c_api/anakin_runner.h $ANAKIN_ROOT/x86_output/anakin_runner.h
fi

if ! [ -e "$ANAKIN_ROOT/x86_output/mklml_include" ]; then
    cp -r $BUILD_OUT_PATH/mklml_include $ANAKIN_ROOT/x86_output/mklml_include
fi

if ! [ -e "$ANAKIN_ROOT/x86_output/libiomp5.so" ]; then
    cp -r $BUILD_OUT_PATH/libiomp5.so $ANAKIN_ROOT/x86_output/libiomp5.so
    cp -r $BUILD_OUT_PATH/libmklml_intel.so $ANAKIN_ROOT/x86_output/libmklml_intel.so
fi

cp $BUILD_OUT_PATH/libanakin*  $SO_PATH/.

echo "tools/release_build/release_x86_one_arch.sh OK"