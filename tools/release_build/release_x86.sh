#!/bin/bash
set -e
THIS_DIR=`dirname $0`
echo $THIS_DIR
for arch in ivybridge broadwell knl haswell
do
#echo $arch
/bin/bash $THIS_DIR/release_x86_one_arch.sh $arch
done