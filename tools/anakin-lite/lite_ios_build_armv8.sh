#!/bin/bash
# This script shows how one can build a anakin for the Android platform using android-tool-chain.

ANAKIN_LITE_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"
echo "-- Anakin lite root dir is: $ANAKIN_LITE_ROOT"

# build the target into build_android.
BUILD_ROOT=$ANAKIN_LITE_ROOT/build-ios-armv8

#if [ -d $BUILD_ROOT ];then
#	rm -rf $BUILD_ROOT
#fi

mkdir -p $BUILD_ROOT
echo "-- Build anakin lite ios into: $BUILD_ROOT"

# Now, actually build the android target.
echo "-- Building anakin lite ..."
cd $BUILD_ROOT

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../../../cmake/ios/ios.toolchain.cmake \
    -DENABLE_DEBUG=NO \
    -DIOS_PLATFORM=iPhoneOS \
    -DUSE_ARMV8=YES \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
	-DUSE_IOS=YES \
	-DUSE_ANDROID=NO \
	-DTARGET_IOS=YES \
    -DUSE_OPENMP=NO \
    -DBUILD_LITE_UNIT_TEST=NO \
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
if [ -d $OUT_DIR/ios_armv8 ];then
	rm -rf $OUT_DIR/ios_armv8
	mkdir -p $OUT_DIR/ios_armv8/include
    mkdir -p $OUT_DIR/ios_armv8/lib
else
    mkdir -p $OUT_DIR/ios_armv8/include
    mkdir -p $OUT_DIR/ios_armv8/lib
fi

cp -r include/ $OUT_DIR/ios_armv8/include
cp -r lib/ $OUT_DIR/ios_armv8/lib