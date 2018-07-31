# Benchmark

## Machine:

>  CPU: `12-core Intel(R) Xeon(R) CPU E5-2620 v2 @2.10GHz`
>  GPU: `Tesla P4`  
>  cuDNN: `v7`  


## Counterpart of anakin  :

The counterpart of **`Anakin`** is the acknowledged high performance inference engine **`NVIDIA TensorRT 3`** ,   The models which TensorRT 3 doesn't support we use the custom plugins  to support.  

## Benchmark Model  

The following convolutional neural networks are tested with both `Anakin` and `TenorRT3`.
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
    1 | 8.85176 | 8.15362
    2 | 15.6517 | 13.8716
    4 | 26.5303 | 21.8478 
    8 | 48.2286 | 40.496 
    32 | 183.994 | 163.035 

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 887 | 648 
    2 | 965 | 733 
    4 | 991 | 810 
    8 | 1067 | 911 
    32 | 1715 | 1325 

    
### <span id = '2'>Yolo </span>  

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 16.4623| 15.3214
    2 | 26.7082| 25.0305 
    4 | 43.2129| 43.4758
    8 | 80.0053 | 80.7645
    32 | 283.352| 311.152

- GPU Memory Used (`MB`)


    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1226  | 1192 
    2 | 1326  | 1269 
    4 | 1435  | 1356 
    8 | 1563  | 1434 
    32 | 2150  | 1633 

### <span id = '3'> Resnet50 </span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 4.26834   |  3.25853 
    2 |  6.2811  |  6.12156 
    4 | 10.1183  | 10.9219
    8 | 18.1395 |   20.323 
    32 | 66.4728 | 83.9934

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 932 | 272 
    2 | 936   | 318 
    4 | 720  | 376 
    8 | 697  | 480 
    32 |  842  | 835 

### <span id = '4'> Resnet101 </span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 7.58234 | 5.66457  
    2 | 11.6014 | 10.9213
    4 | 18.3298 | 19.3987 
    8 | 32.6523 | 37.5575
    32 | 123.114 | 149.089

- GPU Memory Used (`MB)`

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1020   | 420 
    2 | 961   | 467 
    4 | 943  | 503 
    8 | 885  | 606 
    32 | 1048  | 1077 

###  <span id = '5'> MobileNet V1 </span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 45.2189  |  1.39566
    2 |  46.4538  |  2.50698
    4 | 47.8918  | 4.38727
    8 |  52.3636 |  8.21416
    32 | 83.0503 | 31.33

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 516   | 176 
    2 | 524    | 166 
    4 | 497  | 165 
    8 | 508  | 239 
    32 |  628  | 388 

###  <span id = '6'> MobileNet V2</span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 65.4277 | 1.80542
    2 | 66.2048 | 3.85568
    4 | 68.8045 | 6.80921
    8 | 75.64 | 12.6038
    32 | 124.09 | 47.6079

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 341 | 293
    2 | 353 | 301
    4 | 385 | 319
    8 | 421 | 351
    32 | 637 | 551

## How to run those Benchmark models?

> 1. At first, you should parse the caffe model with [`external converter ->`](../docs/Manual/Converter_en.md).
> 2. Switch to *source_root/benchmark/CNN* directory. Use 'mkdir ./models' to create ./models and put anakin models into this file.
> 3. Use command 'sh run.sh', we will create files in logs to save model log with different batch size. Finally, model latency summary will be displayed on the screen.
> 4. If you want to get more detailed information with op time, you can modify CMakeLists.txt with setting `ENABLE_OP_TIMER` to `YES`, then recompile and run. You will find detailed information in  model log file.





