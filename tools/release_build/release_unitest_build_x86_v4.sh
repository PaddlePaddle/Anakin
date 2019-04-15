#!/bin/bash
set -ex
#bash -c "$( curl http://jumbo.baidu.com/install_jumbo.sh )" && source ~/.bashrc
#jumbo install git
export LANG="zh_CN.UTF-8"
##export PATH=/home/public/git-2.17.1/:$PATH
#export PATH=~/.jumbo/bin/git:$PATH
export PATH=/home/public/cmake-3.3.0-Linux-x86_64/bin/:$PATH
export PATH=/home/scmtools/buildkit/cmake/cmake-3.12.3/bin:$PATH
export PATH=/usr/local/bin/:$PATH
export LD_LIBRARY_PATH=//home/scmtools/buildkit/protobuf/protobuf_2.6.1/:$LD_LIBRARY_PATH
export GIT_SSL_NO_VERIFY=1
echo $PATH
echo "git install path"
which git
#git config core.filemode false
echo "git version:"
git --version
# This script shows how one can build a anakin for the x86 platform
ANAKIN_ROOT="$( cd "$(dirname "$0")"/../.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

# build the target into gpu_build.
BUILD_ROOT=$ANAKIN_ROOT/x86_native_build

mkdir -p $BUILD_ROOT
echo "-- Build anakin x86_native into: $BUILD_ROOT"

# Now, actually build the x86 target.
echo "-- Building anakin ..."
cd $BUILD_ROOT


cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
	-DUSE_ARM_PLACE=NO \
	-DUSE_GPU_PLACE=NO \
    -DNVIDIA_GPU=NO \
    -DAMD_GPU=NO \
	-DUSE_X86_PLACE=YES \
	-DUSE_BM_PLACE=NO \
	-DBUILD_WITH_UNIT_TEST=YES \
    -DBUILD_RPC=OFF \
   	-DUSE_PYTHON=OFF \
    -DUSE_GFLAGS=OFF \
	-DENABLE_DEBUG=OFF \
	-DENABLE_VERBOSE_MSG=NO \
	-DENABLE_MIN_DEPENDENCY=YES \
	-DDISABLE_ALL_WARNINGS=YES \
	-DENABLE_NOISY_WARNINGS=NO \
    -DUSE_OPENMP=YES\
	-DBUILD_SHARED=YES\
    -DBAIDU_RPC_ROOT=/opt/brpc \
    -DX86_COMPILE_482=YES\
	-DBUILD_WITH_FRAMEWORK=YES

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" install
else
    make "-j$(nproc)" install
fi

