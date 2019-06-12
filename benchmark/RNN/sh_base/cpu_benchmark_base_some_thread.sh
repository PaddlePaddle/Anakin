#!/bin/bash

set -e
set -x
core_num=$1
shift



core_range='1-'$core_num


echo ${core_range}

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=${core_num}
unset MKL_NUM_THREADS
export MKL_NUM_THREADS=${core_num}

#taskset -c ${core_range} numactl -l $*
taskset -c ${core_range}  $*
