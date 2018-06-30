#!/bin/bash
# This script shows how one can build a anakin for the Android platform using android-tool-chain. 
export ANDROID_NDK=/home/public/android-ndk-r14b
export ARM_PROTOBUF_ROOT=/home/public/arm-android/protobuf

ANAKIN_LITE_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"
echo "-- Anakin lite root dir is: $ANAKIN_LITE_ROOT"

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
BUILD_ROOT=$ANAKIN_LITE_ROOT/build

if [ -d $BUILD_ROOT ];then
	rm -rf $BUILD_ROOT
fi

mkdir -p $BUILD_ROOT
echo "-- Build anakin lite Android into: $BUILD_ROOT"

# Now, actually build the android target.
echo "-- Building anakin lite ..."
cd $BUILD_ROOT

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../../cmake/android/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="armeabi-v7a with NEON" \
	-DANDROID_NATIVE_API_LEVEL=21 \
	-DUSE_ANDROID=YES \
	-DUSE_IOS=NO \
    -DUSE_OMP=YES

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)" && make install
fi

