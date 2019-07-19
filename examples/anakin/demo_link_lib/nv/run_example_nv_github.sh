#!/bin/bash
set -ex
# set compiler path, default is gcc 4.8.2
GXX_COMPILER="/usr/bin/g++"
CUDNN_ROOT="/usr/local/cudnn_7_3_cuda10"
CUDA_ROOT="/usr/local/cuda"
SCRIPT_PATH=`dirname $0`

#compile the demo file
OUTPUT_DIR="${SCRIPT_PATH}/../../../../output/"

if [ ! -d ${OUTPUT_DIR} ];then
echo "can not find ${OUTPUT_DIR}, you must compile anakin first "
exit
fi

if [ ! -d ${CUDNN_ROOT} ];then
echo "can not find ${CUDNN_ROOT}, check this script "
exit
fi

if [ ! -d ${CUDA_ROOT} ];then
echo "can not find ${CUDA_ROOT}, check this script "
exit
fi

if [ -f demo_test_nv ];then
rm demo_test_nv
fi

${GXX_COMPILER} "${SCRIPT_PATH}/demo_test_nv.cpp" -std=c++11 -I"${OUTPUT_DIR}/" -I"${OUTPUT_DIR}/include/" -I"${OUTPUT_DIR}/mklml_include/include/" -I"${CUDA_ROOT}/include/" -I"${CUDNN_ROOT}/include" -L"${OUTPUT_DIR}/" -L"${CUDNN_ROOT}/lib64/" -L"${CUDA_ROOT}/lib64/" -pthread -ldl -lcudart -lcublas -lcurand -lcudnn -lanakin_saber_common -lanakin  -o demo_test_nv

#download public test model
if [ ! -f mobilenet-v2.anakin2.bin ];then
wget --no-check-certificate  https://github.com/qq332982511/Anakin/releases/download/0.1.1/mobilenet-v2.anakin2.bin
fi
# for different environment we export this
export LD_LIBRARY_PATH="${OUTPUT_DIR}:${CUDNN_ROOT}/lib64/:${LD_LIBRARY_PATH}"

# run demo with omp 4 thread
./demo_test_nv mobilenet-v2.anakin2.bin 1