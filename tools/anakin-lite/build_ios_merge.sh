#!/bin/bash
# This script shows how one can build a merged ios lib.
ANAKIN_LITE_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"
echo "-- Anakin lite root dir is: $ANAKIN_LITE_ROOT"

BUILD_ROOT=$ANAKIN_LITE_ROOT
sh lite_ios_build_armv7.sh
sh lite_ios_build_armv8.sh
lipo -create build-ios-armv7/lib/libanakin_lite_static.a build-ios-armv8/lib/libanakin_lite_static.a -output libanakin_lite_static.a
OUT_DIR=$BUILD_ROOT/../../output
if [ -d $OUT_DIR/ios_merge ];then
	rm -rf $OUT_DIR/ios_merge
	mkdir -p $OUT_DIR/ios_merge/include
    mkdir -p $OUT_DIR/ios_merge/lib
else
    mkdir -p $OUT_DIR/ios_merge/include
    mkdir -p $OUT_DIR/ios_merge/lib
fi

cp -r $ANAKIN_LITE_ROOT/build-ios-armv8/include/ $OUT_DIR/ios_merge/include
cp $ANAKIN_LITE_ROOT/libanakin_lite_static.a $OUT_DIR/ios_merge/lib