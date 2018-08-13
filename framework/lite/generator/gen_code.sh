#!/bin/bash

#################################################
#
# Usage: sh gen_code.sh -n -m -o 
#
#################################################
# print help info
help_gen_code() {
	echo "Usage: sh gen_code.sh [-h] [-n MODEL_NAME] [-m MODEL_PATH] [-o OUTPUT_PATH] [-a AOT_MODE] [-d LOG_DEBUG_INFO]"
    echo ""
	echo "	Generating lite code for target model."
	echo ""
	echo "optional arguments:"
    echo ""
	echo " -h help info"
	echo " -n model name used as the name of generating codes."
	echo " -m path to model "
	echo " -o path to save the generating codes."
	echo " -a aot mode: >0: aot mode, generate .h and .cpp; 0: general mode, generate .lite.info and .lite.bin"
	echo " -d debug mode. [ default 0]"
	exit 1
}

# generating code function
gen_code() { 
	if [ $# -lt 5 ]; then
		exit 1
	fi
	mode_name=$1
	mode_path=$2
	out_path=$3
	aot_mode=$4
	debug_mode=$5
	executor="$( cd "$(dirname "$0")"/src ; pwd -P)"/anakin_lite_executer
	$executor $mode_name $mode_path $out_path $aot_mode $debug_mode
}

# get args
if [ $# -lt 5 ]; then
	help_gen_code
	exit 1
fi

mode_name=0
mode_path=0
out_path="./"
aot_mode=1
debug_mode=0
while getopts h:n:m:o:a:d:hold opt
do
	case $opt in
		n) mode_name=$OPTARG;;
		m) mode_path=$OPTARG;;
		o) out_path=$OPTARG;;
		a) aot_mode=$OPTARG;;
		d) debug_mode=$OPTARG;;
		*) help_gen_code;;
	esac
done

echo "User set model name:             $mode_name"
echo "User set model path:  		   $mode_path"
echo "User set out_path:               $out_path"
echo "aot mode:                        $aot_mode"
echo "debug mode:                      $debug_mode"

if [ ! -f $mode_path ];then
	echo "mode_path: $mode_path not exists."
	exit 1
fi

if [ ! -d $out_path ];then
	echo "out path: $out_path not exists."
	exit 1
fi

gen_code $mode_name $mode_path $out_path $aot_mode $debug_mode

rm $out_path/*.tmp
if [ $aot_mode -lt 1 ]; then
    rm $out_path/*.h
    rm $out_path/*.cpp
    rm $out_path/*.w
    rm $out_path/*.info
fi