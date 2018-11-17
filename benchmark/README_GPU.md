# Benchmark

## Machine:

>  CPU: `12-core Intel(R) Xeon(R) CPU E5-2620 v2 @2.10GHz`
>  GPU: `Tesla P4`  
>  cuDNN: `v7`  


## Counterpart of anakin  :

The counterpart of **`Anakin`** is the acknowledged high performance inference engine **`NVIDIA TensorRT 5`** ,   The models which TensorRT 5 doesn't support we use the custom plugins  to support.  

## Benchmark Model  

The following convolutional neural networks are tested with both `Anakin` and `TenorRT5`.
 You can use pretrained caffe model or the model trained by youself.

> Please note that you should transform caffe model or others into anakin model with the help of [`external converter ->`](../docs/Manual/Converter_en.md)


- [Vgg16](#1)   *caffe model can be found [here->](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395)*
- [Yolo](#2)  *caffe model can be found [here->](https://github.com/hojel/caffe-yolo-model)*
- [Resnet50](#3)  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Resnet101](#4)  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Mobilenet v1](#5)  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [Mobilenet v2](#6)  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [RNN](#7)  *not support yet*

We tested them on single-GPU with single-thread. 

### <span id = '1'>VGG16 </span>  

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: |
    1 | 8.53945 | 8.18737
    2 | 14.2269 | 13.8976
    4 | 24.2803 | 21.7976 
    8 | 45.6003 | 40.319  

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1053.88 | 762.73 
    2 | 1055.71 | 762.41 
    4 | 1003.22 | 832.75 
    8 | 1108.77 | 926.9  

    
### <span id = '2'>Yolo </span>  

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 8.41606| 7.07977
    2 | 16.6588| 15.2216 
    4 | 31.9955| 30.5102
    8 | 66.1107 | 64.3658

- GPU Memory Used (`MB`)


    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: |
    1 | 1054.71  | 299.8 
    2 | 951.51  | 347.47 
    4 | 846.9  | 438.47 
    8 | 1042.31  | 515.15  

### <span id = '3'> Resnet50 </span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: |  
    1 | 4.10063   |  3.33845 
    2 |  6.10941  |  5.54814 
    4 | 9.90233  | 10.2763
    8 | 17.3287 |   20.0783 

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1059.15 | 299.86 
    2 | 1077.8   | 340.78 
    4 | 903.04  | 395 
    8 | 832.53  | 508.86  

### <span id = '4'> Resnet101 </span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 7.29828 | 5.672  
    2 | 11.2037 | 9.42352
    4 | 17.9306 | 18.0936 
    8 | 31.4804 | 35.7439

- GPU Memory Used (`MB)`

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1161.94   | 429.22 
    2 | 1190.92   | 531.92 
    4 | 994.11  | 549.7 
    8 | 945.47  | 653.06  

###  <span id = '5'> MobileNet V1 </span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1.52692  |  1.39282
    2 |  1.98091  |  2.05788
    4 | 3.2705  | 4.03476
    8 |  5.15652 |  7.06651

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1144.35   | 99.6 
    2 | 1160.03    | 199.75 
    4 | 1098  | 184.33 
    8 | 990.71  | 232.11  

###  <span id = '6'> MobileNet V2</span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1.95961 | 1.78249
    2 | 2.8709 | 3.01144
    4 | 4.46131 | 5.43946
    8 | 7.161 | 10.2081

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1154.69 | 195.25
    2 | 1187.25 | 227.6
    4 | 1053 | 241.75
    8 | 1062.48 | 352.18

## How to run those Benchmark models?

> 1. At first, you should parse the caffe model with [`external converter ->`](../docs/Manual/Converter_en.md).
> 2. Switch to *source_root/benchmark/CNN* directory. Use 'mkdir ./models' to create ./models and put anakin models into this file.
> 3. Use command 'sh run.sh', we will create files in logs to save model log with different batch size. Finally, model latency summary will be displayed on the screen.
> 4. If you want to get more detailed information with op time, you can modify CMakeLists.txt with setting `ENABLE_OP_TIMER` to `YES`, then recompile and run. You will find detailed information in  model log file.





