#!/bin/bash

#################################################
#
# Usage: sh gen_code.sh -n -m -o
#
#################################################
# print help info
help_gen_code() {
	echo "Usage: sh gen_code.sh [-h] [-n MODEL_NAME] [-m MODEL_PATH] [-p PRECISION_PATH] [-c CALIBRATOR_PATH] [-l LITE_MODE][-o OUTPUT_PATH]\
	 [-a AOT_MODE] [-d LOG_DEBUG_INFO]"
    echo ""
	echo "	Generating lite code for target model."
	echo ""
	echo "optional arguments:"
    echo ""
	echo " -h help info"
	echo " -n model name used as the name of generating codes."
	echo " -m path to model "
	echo " -p path to precision file"
	echo " -c path to calibrator file"
	echo " -l lite model mode. [default 0]"
	echo " -o path to save the generating codes."
	echo " -a aot mode: >0: aot mode, generate .h and .cpp; 0: general mode, generate .lite.info and .lite.bin"
	echo " -d debug mode. [ default 0]"
	echo " -b batch_size. [ default 1]"
	exit 1
}

# generating code function
gen_code() {
	if [ $# -lt 6 ]; then
		exit 1
	fi
	mode_name=$1
	mode_path=$2
	out_path=$3
	aot_mode=$4
	debug_mode=$5
	batch_size=$6
	lite_mode=$7
	prec_path=$8
	cali_path=$9

	executor="$( cd "$(dirname "$0")"/src ; pwd -P)"/anakin_lite_executer
	$executor $mode_name $mode_path $out_path $aot_mode $debug_mode $batch_size $lite_mode $prec_path $cali_path
}

# get args
if [ $# -lt 6 ]; then
	help_gen_code
	exit 1
fi

mode_name=0
mode_path=0
prec_path=0
cali_path=0
lite_mode=0
out_path="./"
aot_mode=1
debug_mode=0
batch_size=1
while getopts h:n:m:p:c:l:o:a:d:b:hold opt
do
	case $opt in
		n) mode_name=$OPTARG;;
		m) mode_path=$OPTARG;;
		p) prec_path=$OPTARG;;
		c) cali_path=$OPTARG;;
		l) lite_mode=$OPTARG;;
		o) out_path=$OPTARG;;
		a) aot_mode=$OPTARG;;
		d) debug_mode=$OPTARG;;
		b) batch_size=$OPTARG;;
		*) help_gen_code;;
	esac
done

echo "User set model name:             $mode_name"
echo "User set model path:  		   $mode_path"
echo "User set out_path:               $out_path"
echo "aot mode:                        $aot_mode"
echo "lite mode:                       $lite_mode"
echo "debug mode:                      $debug_mode"
echo "batch_size:                      $batch_size"


if [ -f $prec_path ];then
	echo "User set precision file path:       $prec_path"
fi

if [ -f $cali_path ];then
	echo "User set calibrator file path:      $cali_path"
fi

if [ ! -f $mode_path ];then
	echo "mode_path: $mode_path not exists."
	exit 1
fi

if [ ! -d $out_path ];then
	echo "out path: $out_path not exists."
	exit 1
fi

if [ $prec_path = 0 ];then
    prec_path=""
fi

if [ $cali_path = 0 ];then
    cali_path=""
fi

gen_code $mode_name $mode_path $out_path $aot_mode $debug_mode $batch_size $lite_mode $prec_path $cali_path

rm $out_path/*.tmp
if [ $aot_mode -lt 1 ]; then
    rm $out_path/*.h
    rm $out_path/*.cpp
fi
