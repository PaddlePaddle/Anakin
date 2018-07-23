# Anakin12

[![Build Status](https://travis-ci.org/PaddlePaddle/Anakin.svg?branch=developing)](https://travis-ci.org/PaddlePaddle/Anakin)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/xklnono/Anakin/badge.svg)](https://coveralls.io/github/xklnono/Anakin)


Welcome to the Anakin GitHub.

Anakin is a cross-platform, high-performance inference engine, which is originally
developed by Baidu engineers and is a large-scale application of industrial products.

Please refer to our [release announcement](https://github.com/PaddlePaddle/Anakin/releases) to track the latest feature of Anakin.

## Features

- **Flexibility**

    Anakin supports a wide range of neural network architectures and
    different hardware platforms. It is easy to run Anakin on GPU / x86 / ARM platform.

-  **High performance**

    In order to give full play to the performance of hardware, we optimized the
    forward prediction at different levels.
      - Automatic graph fusion. The goal of all performance optimizations under a
      given algorithm is to make the ALU as busy as possible. Operator fusion
      can effectively reduce memory access and keep the ALU busy.

      - Memory reuse. Forward prediction is a one-way calculation. We reuse
      the memory between the input and output of different operators, thus
      reducing the overall memory overhead.

      - Assembly level optimization. Saber is a underlying DNN library for Anakin, which
      is deeply optimized at assembly level. Performance comparison between Anakin, TensorRT
      and Tensorflow-lite, please refer to the [benchmark tests](benchmark/README.md).


## Installation

It is recommended to check out the
[docker installation guide](docker/README.md).
before looking into the
[build from source guide](docs/Manual/INSTALL_en.md).

For ARM, please refer [run on arm](docs/Manual/run_on_arm_en.md).

## Benchmark
It is recommended to check out the [readme of benchmark](benchmark/README.md).

## Documentation

We provide [English](docs/Manual/Tutorial_en.md) and [Chinese](docs/Manual/Tutorial_ch.md) documentation.

- Developer guide

  You might want to know more details of Anakin and make it better. Please refer to [how to add custom devices](docs/Manual/addCustomDevice.md) and [how to add custom device operators](docs/Manual/addCustomOp.md).

- User guide

   You can get the working principle of the project, C++ interface description and code examples from [here](docs/Manual/Tutorial_ch.md). You can also learn about the model converter [here](docs/Manual/Converter_ch.md).

- [How to Contribute](docs/Manual/Contribution_ch.md)

   We appreciate your contributions!



## Ask Questions

You are welcome to submit questions and bug reports as [Github Issues](https://github.com/PaddlePaddle/Anakin/issues).

## Copyright and License
Anakin is provided under the [Apache-2.0 license](LICENSE).
