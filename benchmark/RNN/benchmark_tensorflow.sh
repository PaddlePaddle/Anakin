#!/bin/bash
set -e
set -x
sdir=$(cd `dirname $0`; pwd)
$sdir/cpu_benchmark_base_some_thread.sh 1 python $sdir/tensorflow_language_model.py
