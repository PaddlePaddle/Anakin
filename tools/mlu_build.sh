#!/bin/bash
# This script shows how one can build a anakin for the <MLU> platform
ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

# build the target into mlu_build.
BUILD_ROOT=$ANAKIN_ROOT/mlu_build

#export PATH=/usr/local/protobuf-3.4.0/bin:$PATH
#export PATH=/usr/lib/ccache:$PATH
#export CNML_ROOT=$ANAKIN_ROOT/third-party/mlu
#export CNRT_ROOT=$ANAKIN_ROOT/third-party/mlu
#
#export LD_LIBRARY_PATH=$CNML_ROOT/lib:$CNRT_ROOT/lib:ANAKIN_ROOT/mlu_build:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$CNML_ROOT/lib:$CNRT_ROOT/lib:ANAKIN_ROOT/mlu_build:$PWD/third-party/mklml/lib:$LD_LIBRARY_PATH


if [ ! -d "$BUILD_ROOT" ]; then
  mkdir "$BUILD_ROOT"
fi
echo "-- Build anakin mlu into: $BUILD_ROOT"

# Now, actually build the mlu target.
echo "-- Building anakin ..."
cd $BUILD_ROOT

  cmake .. \
  	-DENABLE_DEBUG=NO \
  	-DUSE_MLU_PLACE=YES \
  	-DUSE_BANG=NO \
    -DUSE_OPENCV=NO \
  	-DUSE_ARM_PLACE=NO \
  	-DUSE_GPU_PLACE=NO \
  	-DUSE_NV_GPU=NO \
  	-DUSE_AMD_GPU=NO \
  	-DUSE_X86_PLACE=YES \
  	-DUSE_BM_PLACE=NO \
  	-DBUILD_WITH_UNIT_TEST=YES \
    -DUSE_PYTHON=OFF \
  	-DENABLE_VERBOSE_MSG=NO \
  	-DDISABLE_ALL_WARNINGS=YES \
  	-DENABLE_NOISY_WARNINGS=NO \
    -DUSE_OPENMP=YES\
  	-DBUILD_SHARED=YES\
  	-DBUILD_WITH_FRAMEWORK=YES\
    -DUSE_GFLAGS=NO\
  	-DUSE_BOOST=NO\
    -DBUILD_EXAMPLES=NO

# build target lib or unit test.

if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)" install
fi

