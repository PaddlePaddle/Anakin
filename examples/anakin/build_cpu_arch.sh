#!/bin/bash
FLAG="-I./ -I../../build/ -std=c++11 -ldl -Wno-narrowing "
g++ get_cpu_arch.cpp -o get_cpu_arch $FLAG