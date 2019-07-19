#!/bin/bash
# This script shows how one can build a anakin for the Android platform using android-tool-chain.
set -e

export ANDROID_NDK=/home/public/android-ndk-r16b/

ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

if [ -z "$ANDROID_NDK" ]; then
    echo "-- Did you set ANDROID_NDK variable?"
    exit 1
fi

if [ -d "$ANDROID_NDK" ]; then
    echo "-- Using Android ndk at $ANDROID_NDK"
else
    echo "-- Cannot find ndk: did you install it under $ANDROID_NDK ?"
    exit 1
fi

# build the target into build_android.
BUILD_ROOT=$ANAKIN_ROOT/android_build_armv7

# if [ -d $BUILD_ROOT ];then
#     rm -rf $BUILD_ROOT
# fi

mkdir -p $BUILD_ROOT
echo "-- Build anakin Android into: $BUILD_ROOT"

# Now, actually build the android target.
#../cmake/android/android.toolchain.cmake \
#"armeabi-v7a with NEON" \ "arm64-v8a" \
# -DANDROID_STL=c++_static \
echo "-- Building anakin ..."
cd $BUILD_ROOT
rm -rf *
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="armeabi-v7a with NEON" \
    -DANDROID_TOOLCHAIN=gcc \
    -DANDROID_NATIVE_API_LEVEL=21 \
    -DUSE_ARM_PLACE=YES \
    -DUSE_GPU_PLACE=NO \
    -DUSE_X86_PLACE=NO \
    -DTARGET_ANDROID=YES \
    -DBUILD_WITH_UNIT_TEST=YES \
    -DUSE_PYTHON=OFF \
    -DENABLE_DEBUG=NO \
    -DENABLE_VERBOSE_MSG=NO \
    -DDISABLE_ALL_WARNINGS=YES \
    -DENABLE_NOISY_WARNINGS=NO \
    -DUSE_OPENMP=YES \
    -DENABLE_OP_TIMER=NO \
    -DBUILD_SHARED=NO\
    -DBUILD_EXAMPLES=NO \
    -DBUILD_WITH_FRAMEWORK=YES \
    -DUSE_OPENCV=NO

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make -j4  #&& make install
else
    make -j4  #&& make install
fi

