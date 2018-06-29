#!/bin/bash
sdir=$(cd `dirname $0`; pwd)

if [ ! -e $sdir/model/language_model/language_model.anakin2.bin ]; then
echo "can not find language_model for anakin download now"
wget -P $sdir/model/language_model/ http://ojf1xbmzo.bkt.clouddn.com/language_model.anakin2.bin
fi

if [ ! -e $sdir/data/ptb.valid.txt ]; then
echo "can not find language_data download now"
wget -P $sdir/data/ http://ojf1xbmzo.bkt.clouddn.com/ptb.valid.txt
fi

if [ ! -e $sdir/data/ner_data.txt ]; then
echo "can not find language_data download now"
wget -P $sdir/data/ https://raw.githubusercontent.com/PaddlePaddle/models/develop/fluid/chinese_ner/data/test_files/test_part_1
for n in $(seq 30); do cat $sdir/data/test_part_1 >> $sdir/data/ner_data.txt; done
rm $sdir/data/test_part_1
fi

if [ ! -e $sdir/data/ptb.valid_tokenlize.txt ]; then
python $sdir/read_ptb_data.py
fi


