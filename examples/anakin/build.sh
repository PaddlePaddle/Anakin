#!/bin/bash
DEBUG_FLAG="-std=c++11 -g -I../../framework/c_api/ -I./ -I../../build/  -ldl -Wno-narrowing "
ORI_FAST_FLAG="-std=c++11 -Ofast -ffast-math -I../../framework/c_api/ -I./ -ldl -Wno-narrowing "
STATIC_FAST_FLAG="-std=c++11 -Ofast -ffast-math -I../../output -I./ -ldl -Wno-narrowing -I../../output/framework/c_api/"
FAST_FLAG="-std=c++11 -g -static-libstdc++ --sysroot=/opt/compiler/gcc-4.8.2/ -Wl,-rpath,/opt/compiler/gcc-4.8.2/lib64/ -Wl,-dynamic-linker,/opt/compiler/gcc-4.8.2/lib64/ld-linux-x86-64.so.2  -Ofast -ffast-math -I../../output/framework/c_api/ -I./ -I../../framework/c_api/  -ldl -Wno-narrowing "
g++ example.cpp -o example $FAST_FLAG
g++ map_rnn.cpp -o map_rnn ${FAST_FLAG}