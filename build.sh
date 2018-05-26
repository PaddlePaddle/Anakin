#!/bin/bash
# This script shows how one can use the file in tools
export PATH=/home/scmtools/cmake-3.2.2/bin/:$PATH
export PATH=/opt/compiler/gcc-4.8.2/bin:$PATH
export PATH=/usr/local/bin:$PATH
export CUDNN_ROOT=/home/work/cudnn/cudnn_v7/include/

echo 'build.sh start'
ANAKIN_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

cd tools
echo 'sh gpu_build'
./gpu_build.sh

echo 'build finished!'
