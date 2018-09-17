
#!/bin/bash
# This script shows how one can build a anakin for the <NVIDIA> gpu platform 
ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

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
    -DENABLE_DEBUG=YES \
    -DENABLE_VERBOSE_MSG=NO \
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

