#!/bin/bash
set -e
# install dependency
#jumbo install google-gflags

# This script shows how one can use the file in tools
export PATH=/home/scmtools/cmake-3.2.2/bin/:$PATH
export PATH=/opt/compiler/gcc-4.8.2/bin:$PATH
#export PATH=/usr/local/bin:$PATH
export CUDNN_ROOT=/home/work/cudnn/cudnn_v7/include/
#export BAIDU_RPC_ROOT=/opt/brpc/
#export PROTOBUF_ROOT=/opt

echo 'build.sh start'
ANAKIN_ROOT="$( cd "$(dirname "$0")"/ ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

echo 'sh tools/release_build/release_unitest_build_nv.sh'
bash ./tools/release_build/release_unitest_build_nv.sh
echo "tools/release_build/release_unitest_build_nv.sh [build_success]"
cd $ANAKIN_ROOT/
tar -czf  anakin_release_nv.tar.gz output/*
rm -fr $ANAKIN_ROOT/output

echo 'sh tools/release_build/release_unitest_build_x86.sh'
bash ./tools/release_build/release_unitest_build_x86.sh
echo "tools/release_build/release_unitest_build_x86.sh [build_success]"
cd $ANAKIN_ROOT/
tar -czf  anakin_release_native_x86.tar.gz output/*
rm -fr $ANAKIN_ROOT/output


export ANDROID_NDK=/home/qa_work/arm_tools/android-ndk-r14b
cd $ANAKIN_ROOT
mkdir output
echo 'build.sh start'
echo 'sh build_lite.sh'
bash ./tools/build_lite.sh
echo 'build lite finished!'
#cd tools/anakin-lite
sed -i "s/export ANDROID_NDK=\/home\/public\/android-ndk-r14b/#/g" tools/anakin-lite/lite_android_build_armv7.sh
sed -i "s/export ANDROID_NDK=\/home\/public\/android-ndk-r14b/#/g" tools/anakin-lite/lite_android_build_armv8.sh
echo 'sh lite_android_build_armv7.sh'
bash ./tools/anakin-lite/lite_android_build_armv7.sh
echo 'build lite_android_armv7 finished!'
echo 'sh lite_android_build_armv8.sh'
bash ./tools/anakin-lite/lite_android_build_armv8.sh
echo 'build lite_android_armv8 finished!'

tar -czf anakin_arm_release.tar.gz output/*
rm -fr $ANAKIN_ROOT/output/*

mv $ANAKIN_ROOT/anakin_release_nv.tar.gz $ANAKIN_ROOT/output/
mv $ANAKIN_ROOT/anakin_release_native_x86.tar.gz $ANAKIN_ROOT/output/
mv $ANAKIN_ROOT/anakin_arm_release.tar.gz $ANAKIN_ROOT/output/