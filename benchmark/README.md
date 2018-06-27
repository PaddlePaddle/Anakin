# Benchmark

## Machine:

This time, we only provide benchmark on GPU. In the near future, we will add benchmark on ARM and CPU.

>  CPU: `12-core Intel(R) Xeon(R) CPU E5-2620 v2 @2.10GHz`
>  GPU: `Tesla P4`  
>  cuDNN: `v7`  

## Counterpart of anakin  :
The counterpart of **`Anakin`** is the acknowledged high performance inference engine **`NVIDIA TensorRT 3`** ,   The models which TensorRT 3 doesn't support we use the custom plugins  to support.  

## Benchmark Model  

The following convolutional neural networks are tested with both `Anakin` and `TenorRT3`.
 You can use pretrained caffe model or the model trained by youself.

> Please note that you should transform caffe model or others into anakin model with the help of [`external converter ->`](#)


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
    1 | 8.8690 | 8.2815
    2 | 15.5344 | 13.9116
    4 | 26.6000 | 21.8747 
    8 | 49.8279 | 40.4076 
    32 | 188.6270 | 163.7660 

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 963 | 997
    2 | 965 | 1039
    4 | 991 | 1115
    8 | 1067 | 1269
    32 | 1715 | 2193

    
### <span id = '2'>Yolo </span>  

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 16.4596| 15.2124
    2 | 26.6347| 25.0442 
    4 | 43.3695| 43.5017
    8 | 80.9139 | 80.9880
    32 | 293.8080| 310.8810

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 1569 | 1775
    2 | 1649 | 1815
    4 | 1709 | 1887
    8 | 1731 | 2031
    32 | 2253 | 2907

### <span id = '3'> Resnet50 </span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 4.2459   |  4.1061 
    2 |  6.2627  |  6.5159 
    4 | 10.1277  | 11.3327
    8 | 17.8209 |   20.6680 
    32 | 65.8582 | 77.8858

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 531  | 503
    2 | 543  | 517
    4 | 583 | 541
    8 | 611 | 589
    32 |  809 | 879

### <span id = '4'> Resnet101 </span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 7.5562 | 7.0837  
    2 | 11.6023 | 11.4079
    4 | 18.3650 | 20.0493 
    8 | 32.7632 | 36.0648
    32 | 123.2550 | 135.4880

- GPU Memory Used (`MB)`

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 701  | 683
    2 | 713  | 697
    4 | 793 | 721
    8 | 819 | 769
    32 | 1043 | 1059
 

###  <span id = '5'> MobileNet V1 </span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 45.5156  |  1.3947
    2 |  46.5585  |  2.5483
    4 | 48.4242  | 4.3404
    8 |  52.7957 |  8.1513
    32 | 83.2519 | 31.3178

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 329  | 283
    2 | 345   | 289
    4 | 371 | 299
    8 | 393 | 319
    32 |  531 | 433

###  <span id = '6'> MobileNet V2</span> 

- Latency (`ms`) of different batch  

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 65.6861 | 2.9842
    2 | 66.6814 | 4.7472
    4 | 69.7114 | 7.4163
    8 | 76.1092 | 12.8779
    32 | 124.9810 | 47.2142

- GPU Memory Used (`MB`)

    BatchSize | TensorRT | Anakin
    :---: | :---: | :---: | 
    1 | 341 | 293
    2 | 353 | 301
    4 | 385 | 319
    8 | 421 | 351
    32 | 637 | 551




## How to run those Benchmark models?

> Please refer to [Instructions](CNN/README.md)



# RNN Benchmark \<Anakin VS Tensorflow\>

## Machine:

This time, we only provide benchmark on CPU. In the near future, we will add benchmark on ARM and GPU.

## Counterpart of anakin  :

The counterpart of **`Anakin`** is `Tensorflow 1.8.0`, which installed by Anaconda 4.5.4, run by Python 3.6

## Benchmark Model

The following convolutional neural networks are tested with both `Anakin` and `Tensorflow`.
 You can use pretrained model or the model trained by youself.

> Please note that you should transform fluid model or others into anakin model with the help of [`external converter ->`](#)


- [Language model](#1)   *fluid model can be found [here->](https://github.com/PaddlePaddle/models/tree/develop/fluid/language_model)*


We tested them on single-CPU with different thread numbers.
For language model, we use ptb_valid_txt as dataset.

Tensorflow run in python api, and the thread number for tensorfow is actually process number. You can run it by ` sh benchmark_tensorflow.sh`

Anakin run in c api, and we set openmp thread pool = 1, mkl thread pool=1. You can run it by `sh benchmark_anakin.sh`

All test run in docker 1.13.1, with CentOS Linux release 7.5.1804
### <span id = '1'>language model in i7-7700 </span>

- Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 5.64    | 2.44
    2 | 8.29    | 4.44
    4 | 14.23   | 9.91
    6 | 19.83   | 15.51

- Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 3459 | 8536
    2 | 4772 | 9399
    4 | 5498 | 8418
    6 | 5764 | 8070

### <span id = '2'>language model in E5-2620 v4 </span>

- Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 6.31    | 2.84
    2 | 7.94    | 2.678
    4 | 8.66    | 4.32
    6 | 12.33   | 7.12

- Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 2890 | 7257
    2 | 4726 | 15439
    4 | 8659 | 18351
    6 | 9414 | 17461

### <span id = '3'>language model in E5-2650 v4 </span>

- Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 3.69    | 2.84
    2 | 4.62    | 2.85
    4 | 7.78    | 3.48
    6 | 13.54   | 4.79

- Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 4456 | 7300
    2 | 7522 | 14556
    4 | 9580 | 22086
    6 | 8664 | 23938

### <span id = '4'>text_classfication model in i7-7700 </span>

- Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 1.25    | 0.32
    2 | 1.87    | 0.33
    4 | 2.01   | 0.35
    6 | 2.81   | 0.58

- Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 12797 | 53506
    2 | 17933 | 95898
    4 | 31965 | 148427
    6 | 31784 | 118684
### <span id = '5'>text_classfication in E5-2620 v4</span>

- Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 3.89    | 0.58
    2 | 3.77    | 0.61
    4 | 3.05   | 0.62
    6 | 3.84   | 0.66

- Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 4281 | 28192
    2 | 8804 | 49840
    4 | 19949 | 89710
    6 | 24798 | 116975
### <span id = '6'>text_classfication in E5-2650 v4 </span>

- Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 2.26    | 0.67
    2 | 2.34    | 0.7
    4 | 2.25   | 0.72
    6 | 2.47   | 0.73

- Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 6337 | 24636
    2 | 12266 | 45368
    4 | 24869 | 81952
    6 | 34872 | 109993

### <span id = '7'>chinese_ner model in i7-7700 </span>

- Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 1.96    | 0.094
    2 | 2.59    | 0.098
    4 | 3.74   | 0.1
    6 | 3.95   | 0.13

- Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 8747 | 156564
    2 | 13293 | 208484
    4 | 18294 | 114348
    6 | 25338 | 66480
### <span id = '8'>chinese_ner in E5-2620 v4</span>

- Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 5.44    | 0.13
    2 | 5.45    | 0.14
    4 | 4.84   | 0.15
    6 | 5.18   | 0.16

- Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 4281 | 93527
    2 | 8804 | 127232
    4 | 19949 | 118649
    6 | 24798 | 99553
### <span id = '9'>chinese_ner in E5-2650 v4 </span>

- Latency (`ms`) of one batch

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 3.61    | 0.16
    2 | 3.78    | 0.16
    4 | 3.74   | 0.17
    6 | 3.78   | 0.16

- Throughput (`words/s`)

    ThreadNum | Tensorflow | Anakin
    :---: | :---: | :---: |
    1 | 4669 | 79225
    2 | 8953 | 115761
    4 | 18074 | 118696
    6 | 26607 | 102044

## How to run those Benchmark models?


> Please refer to [Instructions](RNN/README.md)

# RNN Benchmark \<Anakin VS PaddlePaddle/Fluid\>

## Machine:

This time, we only provide benchmark on CPU. In the near future, we will add benchmark on ARM and GPU.

## Counterpart of anakin  :

The counterpart of **`Anakin`** is `Fluid`,commit = 0b3d7f1f4c525f40cc178774e0eec74f88047cc9

## Benchmark Model

The following convolutional neural networks are tested with both `Anakin` and `Fluid`.
 You can use pretrained model or the model trained by youself.

> Please note that you should transform fluid model or others into anakin model with the help of [`external converter ->`](#)


- [Language model](#1)   *fluid model can be found [here->](https://github.com/PaddlePaddle/models/tree/develop/fluid/language_model)*

- [Chinese_ner](#2)   *fluid model can be found [here->](https://github.com/PaddlePaddle/models/blob/develop/fluid/chinese_ner)*

We tested them on single-CPU with different thread numbers.

Anakin and Fluid run in c api, and we set openmp thread pool = 1, mkl thread pool=1.

### <span id = '1'>language model in E5-2650 v4 </span>

- Latency (`ms`) of one batch

    ThreadNum | Fluid | Anakin
    :---: | :---: | :---: |
    1 | 42.09    | 1.90
    2 | 42.14    | 2.16
    6 | 42.15   | 4.21
    10 | 42.14   | 9.26
    12 | 42.34   | 11.17

- Throughput (`sentence/s`)

    ThreadNum | Fluid | Anakin
    :---: | :---: | :---: |
    1 | 23 | 524
    2 | 47 | 916
    6 | 141 | 1402
    10 | 236   | 1063
    12 | 282   | 1044

### <span id = '2'>Chinese_ner model in E5-2650 v4 </span>

- Latency (`ms`) of one batch

    ThreadNum | Fluid | Anakin
    :---: | :---: | :---: |
    1 | 0.47    | 0.17
    4 | 0.26    | 0.17
    6 | 0.36    | 0.17
    10 | 0.59   | 0.17
    12 | 0.72   | 0.17

- Throughput (`sentence/s`)

    ThreadNum | Fluid | Anakin
    :---: | :---: | :---: |
    1 | 2129  | 5819
    4 | 3866  | 11182
    6 | 8095  | 30948
    10 | 8250 | 44093
    12 | 8112  | 47185