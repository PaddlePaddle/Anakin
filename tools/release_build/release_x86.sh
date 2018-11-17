#!/bin/bash
set -e
# This script shows how one can build a anakin for the <X86> platform
ANAKIN_ROOT="$( cd "$(dirname "$0")"/../.. ; pwd -P)"
echo "-- Anakin root dir is: $ANAKIN_ROOT"

THIS_DIR=`dirname $0`
echo $THIS_DIR
for arch in ivybridge broadwell knl
do
#echo $arch
/bin/bash $THIS_DIR/release_x86_one_arch.sh $arch
done
echo "compile OK"

cd $ANAKIN_ROOT
tar -czf anakin_x86_release.tar.gz x86_output
tar -czf anakin_example.tar.gz examples
echo "compress OK"