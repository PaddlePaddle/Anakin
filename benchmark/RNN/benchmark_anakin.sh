#!/bin/bash
set -e
set -x
sdir=$(cd `dirname $0`; pwd)

sh $sdir/prepare.sh

#sh $sdir/sh_base/cpu_benchmark_base_some_thread.sh 1 $sdir/../../output/unit_test/net_exec_x86_oneinput $sdir/model/language_model/ $sdir/data/ptb.valid_tokenlize.txt 1
#sh $sdir/sh_base/cpu_benchmark_base_some_thread.sh 2 $sdir/../../output/unit_test/net_exec_x86_oneinput $sdir/model/language_model/ $sdir/data/ptb.valid_tokenlize.txt 2
#sh $sdir/sh_base/cpu_benchmark_base_some_thread.sh 4 $sdir/../../output/unit_test/net_exec_x86_oneinput $sdir/model/language_model/ $sdir/data/ptb.valid_tokenlize.txt 4
#sh $sdir/sh_base/cpu_benchmark_base_some_thread.sh 6 $sdir/../../output/unit_test/net_exec_x86_oneinput $sdir/model/language_model/ $sdir/data/ptb.valid_tokenlize.txt 6

for i in {1,2,4,6} ;do
$sdir/../../output/unit_test/net_exec_x86_oneinput $sdir/model/language_model/ $sdir/data/ptb.valid_tokenlize.txt $i
done

for i in {1,2,4,6} ;do
$sdir/../../output/unit_test/net_exec_test_chinese_ner $sdir/model/chinese_ner_model/ $sdir/data/ner_data.txt $i 1
done

for i in {1,2,4,6} ;do
$sdir/../../output/unit_test/net_exec_x86_oneinput $sdir/model/text_classfication/ $sdir/data/ptb.valid_tokenlize.txt $i
done