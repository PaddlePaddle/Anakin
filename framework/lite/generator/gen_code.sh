#!/bin/bash

#################################################
#
# Usage: sh gen_code.sh -n -m -o 
#
#################################################
# print help info
help_gen_code() {
	echo "Usage: sh gen_code.sh [-h] [-n MODEL_NAME] [-m MODEL_PATH] [-o OUTPUT_PATH]"
    echo ""
	echo "	Generating lite code for target model."
	echo ""
	echo "optional arguments:"
    echo ""
	echo " -h help info"
	echo " -n model name used as the name of generating codes."
	echo " -m path to model "
	echo " -o path to save the generating codes. [ default './']"
	echo " -d debug mode. [ default 0]"
	exit 1
}

# generating code function
gen_code() { 
	if [ $# -lt 4 ]; then
		exit 1
	fi
	mode_name=$1
	mode_path=$2
	out_path=$3
	debug_mode=$4
	executor="$( cd "$(dirname "$0")"/src ; pwd -P)"/anakin_lite_executer
	$executor $mode_name $mode_path $out_path $debug_mode
}

# get args
if [ $# -lt 4 ]; then
	help_gen_code
	exit 1
fi

mode_name=0
mode_path=0
out_path="./"
debug_mode=0
while getopts h:n:m:o:d:hold opt
do
	case $opt in
		n) mode_name=$OPTARG;;
		m) mode_path=$OPTARG;;
		o) out_path=$OPTARG;;
		d) debug_mode=$OPTARG;;
		*) help_gen_code;;
	esac
done

echo "User set model name:             $mode_name"
echo "User set model path:  		   $mode_path"
echo "User set out_path:               $out_path"
echo "debug mode:                      $debug_mode"

if [ ! -f $mode_path ];then
	echo "mode_path: $mode_path not exists."
	exit 1
fi

if [ ! -d $out_path ];then
	echo "out path: $out_path not exists."
	exit 1
fi

gen_code $mode_name $mode_path $out_path $debug_mode

rm $out_path/*.tmp