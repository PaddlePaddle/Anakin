# Anakin2.0
---

## Benchmark

这里您可以看到Anakin与业界其它框架的性能对比，其中GPU对比TensorRT，CPU对比Intel，arm对比NCNN.

>Anakin realease [性能列表](../benchmark/README.md)

## API reference

这里您可以看到Anakin的工作原理和相关接口的介绍，方便开发者使用和进行二次开发,同时我们还提供依赖发布产出的详细用户手册。

>Anakin [数据类型](Manual/C++APIs_ch.md)

>Ankain [Api](Manual/op.md)

>详细[用户手册](Manual/Anakin_helper_ch.md)

## 安装Anakin

您可以通过编译Anakin源码安装，也可以通过docker安装。

>[docker安装](../docker/README_cn.md)

>[源码安装](Manual/INSTALL_ch.md)

## 快速上手

我们为您提供了各个平台上面的快速上手说明，里面有简单的使用说明，您可以很快自己写一个[Anakin Inference demo](Manual/QuickStart_ch.md)。

>[arm](Manual/run_on_arm_ch.md)

>[GPU](Manual/QuickStart_ch.md)

>[CPU](Manual/QuickStart_ch.md)

>[寒武纪](Manual/INSTALL_ch.md)

>[比特大陆](Manual/INSTALL_ch.md)

>[模型解析](Manual/Converter_ch.md)


## 完整示例

在这里我们为您提供了各个平台上面的完整示例，您可以很快完整一个模型的性能实验。

>[arm](../examples/arm)

>[cuda](../examples/cuda)

>[x86](../examples/x86)

>[寒武纪](../examples/mlu)

>[比特大陆](../examples/bitmain)

>[集成Anakin](../examples/anakin)

>[arm_lite](../examples/anakin/arm_lite)


## 扩展
为了方便开发者自己添加新设备和添加kernel，我们还为您提供了相应的原理介绍和示例说明。

>添加[新设备](Manual/addCustomDevice.md)

>添加[kernel](Manual/addCustomOp.md)
