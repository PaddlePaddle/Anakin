#!/bin/bash
set -e
# install dependency
#jumbo install google-gflags

# This script shows how one can use the file in tools
#export PATH=/home/public/git-2.17.1/:$PATH
export LANG="zh_CN.UTF-8"
export PATH=/home/scmtools/cmake-3.2.2/bin/:$PATH
#export PATH=/home/users/lizhenguo/.jumbo/bin/:$PATH
#export PATH=/home/public/cmake-3.3.0-Linux-x86_64/bin/:$PATH
export PATH=/opt/compiler/gcc-4.8.2/bin:$PATH
#export PATH=/usr/local/bin/:$PATH
#export LD_LIBRARY_PATH=//home/scmtools/buildkit/protobuf/protobuf_2.6.1/:$LD_LIBRARY_PATH
#export GIT_SSL_NO_VERIFY=1
#echo $PATH
#echo "git install path"
#which git
#echo "git version:"
#git --version
#export PATH=/usr/local/bin:$PATH
export CUDNN_ROOT=/home/work/cudnn/cudnn_v7/include/
#export BAIDU_RPC_ROOT=/opt/brpc/
#export PROTOBUF_ROOT=/opt

echo 'build.sh start'
ANAKIN_ROOT="$( cd "$(dirname "$0")"/ ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

echo "device is %s $1"

if [ $1 == "nv" ]; then
    echo "combile device is GPU"
    echo 'sh tools/release_build/release_unitest_build_nv.sh'
    bash ./tools/release_build/release_unitest_build_nv.sh
    echo "tools/release_build/release_unitest_build_nv.sh [build_success]"
    cd $ANAKIN_ROOT/
    cp -r tools/external_converter_v2 output/
    tar -czf  anakin_release_nv.tar.gz output/*
    rm -rf $ANAKIN_ROOT/output/*
    mkdir -p output/gpucpuv2
    mv anakin_release_nv.tar.gz output/gpucpuv2/
elif [ $1 == "x86_v4" ]; then
    echo " combile device is X86_v4"
    echo 'sh tools/release_build/release_unitest_build_x86_v4.sh'
    bash ./tools/release_build/release_unitest_build_x86_v4.sh
    echo "tools/release_build/release_unitest_build_x86_v4.sh [build_success]"
    cd $ANAKIN_ROOT/
    #mkdir output
    cp -r tools/external_converter_v2 output/
    tar -czf  anakin_release_native_x86_v4.tar.gz output/*
    rm -rf $ANAKIN_ROOT/output/*
    mkdir -p output/cpuv4
    mv anakin_release_native_x86_v4.tar.gz output/cpuv4/
elif [ $1 == "x86_v5" ]; then
    echo " combile device is X86_v5"
    echo 'sh tools/release_build/release_unitest_build_x86_v5.sh'
    bash ./tools/release_build/release_unitest_build_x86_v5.sh
    echo "tools/release_build/release_unitest_build_x86_v5.sh [build_success]"
    cd $ANAKIN_ROOT/
    #mkdir output
    cp -r tools/external_converter_v2 output/
    tar -czf  anakin_release_native_x86_v5.tar.gz output/*
    rm -rf $ANAKIN_ROOT/output/*
    mkdir -p output/cpuv5
    mv anakin_release_native_x86_v5.tar.gz output/cpuv4/
elif [ $1 == "arm" ]; then
    echo "combile device is arm"
    export ANDROID_NDK=/home/qa_work/arm_tools/android-ndk-r16b
    export PATH=/usr/local/bin/:$PATH
    export CMAKE_ROOT=/usr/local/bin/
    cd $ANAKIN_ROOT
    #mkdir -p output_arm/arm
    #mv output/* output_arm/
    echo 'build.sh start'
    echo 'sh build_lite.sh'
    bash ./tools/build_lite.sh
    echo 'build lite finished!'
    #cd tools/anakin-lite
    sed -i "s/export ANDROID_NDK=\/home\/public\/android-ndk-r16b/#/g" tools/anakin-lite/lite_android_build_armv7.sh
    sed -i "s/export ANDROID_NDK=\/home\/public\/android-ndk-r16b/#/g" tools/anakin-lite/lite_android_build_armv8.sh
    sed -i "s/export ANDROID_NDK=\/home\/public\/android-ndk-r16b/#/g" tools/anakin-lite/lite_android_build_armv8_clang.sh
    echo 'sh lite_android_build_armv7.sh'
    bash ./tools/anakin-lite/lite_android_build_armv7.sh
    echo 'build lite_android_armv7 finished!'
    echo 'sh lite_android_build_armv8.sh'
    bash ./tools/anakin-lite/lite_android_build_armv8.sh
    echo 'build lite_android_armv8 finished!'
    echo 'sh lite_android_build_armv8_clang.sh'
    bash ./tools/anakin-lite/lite_android_build_armv8_clang.sh
    echo 'build lite_android_armv8_clang finished!'
    cp -r tools/external_converter_v2 output/
    tar -czf  anakin_release_arm.tar.gz output/*
    rm -rf $ANAKIN_ROOT/output/*
    mkdir -p output/arm
    mv anakin_release_arm.tar.gz output/arm/
    ##mv anakin_release_arm.tar.gz output_arm/arm/
    #mv output_arm output
elif [ $1 == "ios" ]; then
    echo "combile device is ios"
    #export ANDROID_NDK=/home/qa_work/arm_tools/android-ndk-r14b
    cd $ANAKIN_ROOT
    mkdir output
    echo 'build.sh start'
    #cd tools/anakin-lite
    sed -i "s/export ANDROID_NDK=\/home\/public\/android-ndk-r16b/#/g" tools/anakin-lite/lite_android_build_armv7.sh
    sed -i "s/export ANDROID_NDK=\/home\/public\/android-ndk-r16b/#/g" tools/anakin-lite/lite_android_build_armv8.sh
    echo 'sh build_ios_merge.sh'
    bash ./tools/anakin-lite/build_ios_merge.sh
    echo 'build lite_ios finished!'
    cp -r tools/anakin-lite output/anakin-lite
    tar -czf anakin_ios_release.tar.gz output/*
    rm -rf $ANAKIN_ROOT/output/*
    x#mkdir -p output/mac
    mkdir -p output/
    mv anakin_ios_release.tar.gz output/
else
    echo "not support $1 device, only support nv, x86_v4, arm and ios"
fi
#mv $ANAKIN_ROOT/anakin_release_nv.tar.gz $ANAKIN_ROOT/output/
#mv $ANAKIN_ROOT/anakin_release_native_x86.tar.gz $ANAKIN_ROOT/output/
#mv $ANAKIN_ROOT/anakin_arm_release.tar.gz $ANAKIN_ROOT/output/
