#!/bin/bash
set -e
set -x

sdir=$(cd `dirname $0`; pwd)

sh $sdir/prepare.sh

#sh $sdir/sh_base/cpu_benchmark_base_some_thread.sh 1 python $sdir/tensorflow_language_model.py 1
#sh $sdir/sh_base/cpu_benchmark_base_some_thread.sh 2 python $sdir/tensorflow_language_model.py 2
#sh $sdir/sh_base/cpu_benchmark_base_some_thread.sh 4 python $sdir/tensorflow_language_model.py 4
#sh $sdir/sh_base/cpu_benchmark_base_some_thread.sh 6 python $sdir/tensorflow_language_model.py 6

for i in {1,2,4,6};do
python $sdir/tensorflow_language_model.py $i
done