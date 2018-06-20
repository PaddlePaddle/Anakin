#!/bin/bash
set -e
set -x
sdir=$(cd `dirname $0`; pwd)

$sdir/cpu_benchmark_base_one_socket.sh