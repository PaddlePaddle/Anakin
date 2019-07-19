#!/bin/bash
# This script shows how one can build protobuf for the Android platform using android-tool-chain.
# IMPORTANT!!!!!!!!!!!!!!
# remove "-g" compile flags in  "$ANDROID_NDK/build/cmake/android.toolchain.cmake"
# to remove debug info
# set your ndk path to ANDROID_NDK
# NDK version is up to r16b, the latest version(r18b) remove gcc from toolchain
# firstly, download the release version of protobuf or git clone the protobuf project, recoment version v3.5.0
# copy this script to protobuf_path/cmake/
# run this script by: sh build_android_protobuf_gcc_armv8.sh
set -e
export ANDROID_NDK=/home/public/android-ndk-r16b

protobuf_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"
echo "-- protobuf root dir is: $protobuf_ROOT"

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

# remove protoc in CMakeList.txt and install.cmake
sed -i "s/include(libprotoc.cmake)/#/g" CMakeLists.txt
sed -i "s/include(protoc.cmake)/#/g" CMakeLists.txt
sed -i "s/libprotoc)/)/g" install.cmake
sed -i "s/install(TARGETS protoc EXPORT protobuf-targets/#/g" install.cmake
sed -i "s/RUNTIME DESTINATION \${CMAKE_INSTALL_BINDIR} COMPONENT protoc)/#/g" install.cmake
sed -i "s/export(TARGETS libprotobuf-lite libprotobuf libprotoc protoc/export(TARGETS libprotobuf-lite libprotobuf/g" install.cmake

# build the target into build_android.
BUILD_ROOT=$protobuf_ROOT/build-protobuf-android-v8-gcc
mkdir -p $BUILD_ROOT
echo "-- Build protobuf Android into: $BUILD_ROOT"

# Now, actually build the android target.
echo "-- Building anakin lite ..."
cd $BUILD_ROOT
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK \
    -DANDROID_NATIVE_API_LEVEL=21 \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_TOOLCHAIN=gcc \
    -DCMAKE_BUILD_TYPE=Release \
    -Dprotobuf_BUILD_EXAMPLES=OFF \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -Dprotobuf_BUILD_STATIC_LIBS=ON \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=$BUILD_ROOT \
    -DANDROID_LINKER_FLAGS="-landroid -llog" \
    -DANDROID_CPP_FEATURES="rtti exceptions" \


# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)" && make install
fi
