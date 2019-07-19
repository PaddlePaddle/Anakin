#!/bin/bash
set -ex
# set compiler path, default is system gcc
GXX_COMPILER="/usr/bin/g++"
SCRIPT_PATH=`dirname $0`
OUTPUT_DIR="${SCRIPT_PATH}/../../../../output/"


if [ ! -d ${OUTPUT_DIR} ];then
echo "can not find ${OUTPUT_DIR}, you must compile anakin and make install first, or modify this script "
exit
fi

if [ ! -f ${GXX_COMPILER} ];then
echo "can not find compiler ${GXX_COMPILER}, you must have compiler firstly or modify this script "
exit
fi


#compile the demo file
if [ -f demo_test_nv ];then
rm demo_test_x86
fi
${GXX_COMPILER} "${SCRIPT_PATH}/demo_test_x86.cpp" -std=c++11 -I"${OUTPUT_DIR}/" -I"${OUTPUT_DIR}/include/" -I"${OUTPUT_DIR}/mklml_include/" -L"${OUTPUT_DIR}/"  -lpthread -ldl -liomp5 -lmkldnn -lmklml_intel -lanakin_saber_common -lanakin  -o demo_test_x86

#download public test model
if [ ! -f mobilenet-v2.anakin2.bin ];then
wget --no-check-certificate  https://github.com/qq332982511/Anakin/releases/download/0.1.1/mobilenet-v2.anakin2.bin
fi

# for different environment we export this
export LD_LIBRARY_PATH="${OUTPUT_DIR}:${LD_LIBRARY_PATH}"

# run demo with omp 4 thread, more thread will get more performance
./demo_test_x86 mobilenet-v2.anakin2.bin 1 4