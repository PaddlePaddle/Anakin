#!/bin/bash
sdir=$(cd `dirname $0`; pwd)

#if [ ! -e $sdir/data/ptb.valid.txt ]; then
#echo "can not find language_data download now"
#wget -P $sdir/data/ http://ojf1xbmzo.bkt.clouddn.com/ptb.valid.txt
#fi

if [ ! -e $sdir/data/ner_data.txt ]; then
echo "can not find language_data download now"
wget -P $sdir/data/ https://raw.githubusercontent.com/PaddlePaddle/models/v0.15.0-rc0/fluid/chinese_ner/data/test_files/test_part_1
for n in $(seq 30); do cat $sdir/data/test_part_1 >> $sdir/data/ner_data.txt; done
rm $sdir/data/test_part_1
fi

if [ ! -e $sdir/data/ptb.valid_tokenlize.txt ]; then
python $sdir/read_ptb_data.py
fi


