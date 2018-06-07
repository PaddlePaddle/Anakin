#!/usr/bin

set -e
if [ ! -d "logs" ]; then
  mkdir logs
fi
function test() {
  model_dir=$1
  model_file=$2
  num=$3
  prefix=$4
  mkdir -p logs/$prefix
  ../../output/unit_test/benchmark \
                        $model_dir \
                        $model_file \
                        $num  \
                        10 \
                        1000 >&logs/$prefix/$prefix-gpu-$num.log
}

spin='-\|/'

model_dir=./models/

model_list=$(ls $model_dir)

for file_name in $model_list
do
    prefix=`basename $file_name .anakin.bin`
    for batch_size in  1 2 4 8 32
    do
        test $model_dir $file_name $batch_size $prefix
        pid=$!
        while kill -0 $pid 2>/dev/null
        do 
            i=$(( (i+1) %4 ))
            printf "\r + running model $file_name BatchSize($batch_size)... ${spin:$i:1}"
            sleep .1
        done
        printf "\r + running model $file_name BatchSize($batch_size) over! \n"
    done
done

#result summary
logs_list=$(ls ./logs/)
for log_file in $logs_list
do
    grep "average time" ./logs/$log_file/* | awk -F '[ ,/]' 'BEGIN {
    printf "MODEL    BATCHSIZE   TIME    SPEED_RATE\n"
    printf "---------------------------------------\n"
    t=0
    }
    {if(NR==1){t=$(NF-1)} 
     printf("%-20s %-8d %.6f  %.6f\n", $(NF-6), $(NF-4), $(NF-1), $(NF-1)/t)}'
done

