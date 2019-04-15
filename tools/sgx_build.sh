#!/bin/bash
# This script shows how one can build a anakin for the <X86> platform,
# with sepcial support for running in SGX mode
ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

# build the target into sgx_build.
BUILD_ROOT=$ANAKIN_ROOT/sgx_build

mkdir -p $BUILD_ROOT
echo "-- Build anakin sgx into: $BUILD_ROOT"

# Now, actually build the gpu target.
echo "-- Building anakin ..."
cd $BUILD_ROOT

cmake .. \
       -DCMAKE_BUILD_TYPE=Release \
       -DUSE_ARM_PLACE=NO \
       -DUSE_GPU_PLACE=NO \
       -DUSE_X86_PLACE=YES \
       -DUSE_SGX=YES \
       -DBUILD_WITH_UNIT_TEST=NO \
       -DUSE_PYTHON=OFF \
       -DENABLE_DEBUG=NO \
       -DENABLE_VERBOSE_MSG=NO \
       -DDISABLE_ALL_WARNINGS=YES \
       -DENABLE_NOISY_WARNINGS=NO 

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)"   && make install
fi

