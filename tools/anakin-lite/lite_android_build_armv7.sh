#!/bin/bash
# This script shows how one can build a anakin for the Android platform using android-tool-chain.
# IMPORTANT!!!!!!!!!!!!!!
# remove "-g" compile flags in  "$ANDROID_NDK/build/cmake/android.toolchain.cmake"
# to remove debug info
export ANDROID_NDK=/home/public/android-ndk-r14b/

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
BUILD_ROOT=$ANAKIN_LITE_ROOT/build-android-v7

#if [ -d $BUILD_ROOT ];then
#	rm -rf $BUILD_ROOT
#fi

mkdir -p $BUILD_ROOT
echo "-- Build anakin lite Android into: $BUILD_ROOT"

# Now, actually build the android target.
echo "-- Building anakin lite ..."
cd $BUILD_ROOT
#-DCMAKE_TOOLCHAIN_FILE=../../../cmake/android/android.toolchain.cmake \ # set toolchain file to file in this project
#-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \ # set toolchain file to NDK default
#-DANDROID_STL=gnustl_static \ # set stl lib
#-DANDROID_TOOLCHAIN=clang \ # set compile to gcc or clang
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../../../cmake/android/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK \
    -DANDROID_NATIVE_API_LEVEL=19 \
    -DANDROID_ABI="armeabi-v7a with NEON" \
    -DENABLE_DEBUG=NO \
    -DUSE_ARMV8=NO \
	-DUSE_ANDROID=YES \
	-DTARGET_IOS=NO \
    -DUSE_OPENMP=YES \
    -DBUILD_LITE_UNIT_TEST=YES \
    -DUSE_OPENCV=NO \
    -DENABLE_OP_TIMER=NO \
    -DUSE_ANDROID_LOG=NO

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)" && make install
fi

OUT_DIR=$BUILD_ROOT/../../../output
if [ -d $OUT_DIR/android_armv7 ];then
	rm -rf $OUT_DIR/android_armv7
	mkdir -p $OUT_DIR/android_armv7/include
    mkdir -p $OUT_DIR/android_armv7/lib
else
    mkdir -p $OUT_DIR/android_armv7/include
    mkdir -p $OUT_DIR/android_armv7/lib
fi

cp -r include/ $OUT_DIR/android_armv7/include
cp -r lib/ $OUT_DIR/android_armv7/lib
cp -r unit_test/ $OUT_DIR/android_armv7/unit_test