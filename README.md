# Anakin


[![Build Status]()]()
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)]()
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)]()
[![Coverage Status]()]()
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


Welcome to the Anakin GitHub.

Anakin is an cross-platform, high-performance inference engine, which is originally
developed by Baidu engineers and is a large-scale application of industrial products.

Please refer to our [release announcement]() to track the latest feature of Anakin.

## Features

- **Flexibility**

    Anakin supports a wide range of neural network architectures and
    diffrent hardware platform. It is easy to run Anakin at GPU/x86/ARM platform.

-  **High performance**

    In order to giving full play to the performance of hardware, we optimize the
    forward prediction at diffrent levels.
      - Automatic graph fusion. The goal of all performance optimization under a 
      given algorithm is to make ALU as busy as possible, Operator fusion 
      can effectively reduce memory access and keep ALU busy.
      
      - Memory reuse. Forward prediction is a one-way calculation. We reuse 
      the memory between the input and output of different operators, thus 
      reducing the overall memory overhead.

      - Assembly level optimization. Saber is Anakin's underlying DNN library, which
      is deeply optimized at assembly level. Performance comparison between Anakin, 
      TensorRT, NCNN, Tensorflow-lite, please refer to the [Benchmark Readme]()



## Installation

It is recommended to check out the
[Docker installation guide]()
before looking into the
[build from source guide]().

## Benchmark
It is recommended to check out the [Benchmark Readme]()

## Documentation

We provide [English]() and
[Chinese]() documentation.

- [Anakin developer guide]()

  You might want to know more details of Anakin and make it better.

- [C++ API]()

   Python API is under-developing.

- [How to Contribute]()

   We appreciate your contributions!


## Ask Questions

You are welcome to submit questions and bug reports as [Github Issues]().

## Copyright and License
Anakin is provided under the [Apache-2.0 license](LICENSE).

