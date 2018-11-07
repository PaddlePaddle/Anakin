#!/bin/bash
FLAG="-g -I../../framework/c_api/ -I./ -std=c++11 -ldl -Wno-narrowing "
g++ example.cpp -o example $FLAG
g++ map_rnn.cpp -o map_rnn $FLAG