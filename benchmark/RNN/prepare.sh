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

if [ ! -e $sdir/data/ptb.valid_tokenlize.txt ]; then
python $sdir/read_ptb_data.py
fi


