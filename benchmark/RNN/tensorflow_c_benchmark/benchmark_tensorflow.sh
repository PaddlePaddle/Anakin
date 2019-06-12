#!/bin/bash
set -e
set -x

sdir=$(cd `dirname $0`; pwd)


for i in {1,2,4,6};do
bazel run //tensorflow/cc:example_model /root/tf_mount/RNN/model/language_model_tf/all.pb /root/tf_mount/RNN/data/ptb.valid_tokenlize.txt  $i
done

for i in {1,2,4,6};do
bazel run //tensorflow/cc:example_model /root/tf_mount/RNN/model/text_classfi_model_tf/all.pb /root/tf_mount/RNN/data/ptb.valid_tokenlize.txt  $i
done
