#!/bin/bash

set -e
core_per_socker=`lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}' | sed 's/^ *\| *$//g'`
core_num=$core_per_socker

echo $core_num
core_idx=$[$core_num-1]
echo $core_idx
core_range='0-'${core_idx}

echo ${core_range}

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=${core_num}
unset MKL_NUM_THREADS
export MKL_NUM_THREADS=${core_num}

taskset -c ${core_range} numactl -l $*