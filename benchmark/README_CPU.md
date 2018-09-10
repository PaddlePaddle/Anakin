# Benchmark 

## Machine:

This time, we only provide benchmark on CPU. In the near future, we will add benchmark on ARM and GPU.

> System: `CentOS 7 in Docker`, for benchmark between Anakin and Tensorflow  
> System: `CentOS 6.3`, for benchmark between Anakin and Paddle

## Counterpart of anakin  :

The counterpart of **`Anakin`** is `Tensorflow 1.8.0`, which installed by Anaconda 4.5.4, run by Python 3.6

## Benchmark Model

 You can use pretrained model or the model trained by youself.

> Please note that you should transform fluid model or others into anakin model with the help of [`external converter ->`](../docs/Manual/Converter_en.md)

- [Language model](#1)   *fluid model can be found [here->](https://github.com/PaddlePaddle/models/tree/develop/fluid/language_model)*
- [Chinese_ner](#4)   *fluid model can be found [here->](https://github.com/PaddlePaddle/models/blob/develop/fluid/chinese_ner)*
- [text_classification](#7)   *fluid model can be found [here->](https://github.com/PaddlePaddle/models/blob/develop/fluid/text_classification)*

We tested them on single-CPU with different thread numbers.

1. **`Anakin`** VS **`Tensorflow`**

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

> 1. You can just run `sh benchmark_tensorflow.sh` and  `sh benchmark_anakin.sh`  
> 2. Get the model of caffe or fluid, convert model to anakin model, use net_test_*** to test your model.  


