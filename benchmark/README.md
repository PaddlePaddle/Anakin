# Benchmark

## Benchmark Model  

> Please note that you should transform caffe model or others into anakin model with the help of [`external converter ->`](#)

### GPU 

The following convolutional neural networks are tested with both `Anakin` and `TenorRT3` on GPU.
 You can use pretrained caffe model or the model trained by youself.

- [Vgg16]()  *caffe model can be found [here->](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395)*
- [Yolo]()  *caffe model can be found [here->](https://github.com/hojel/caffe-yolo-model)*
- [Resnet50]()  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Resnet101]()  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Mobilenet v1]()  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [Mobilenet v2]()  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [RNN]()  *not support yet*

### CPU

The following convolutional neural networks are tested with `Anakin`, 'Tensorflow' and `Tensorflow`.
 You can use pretrained model or the model trained by youself.

- [Language model]()   *fluid model can be found [here->](https://github.com/PaddlePaddle/models/tree/develop/fluid/language_model)*
- [Chinese_ner]()   *fluid model can be found [here->](https://github.com/PaddlePaddle/models/blob/develop/fluid/chinese_ner)*
- [text_classification]()   *fluid model can be found [here->](https://github.com/PaddlePaddle/models/blob/develop/fluid/text_classification)*

### ARM

The following convolutional neural networks are tested with `Anakin`, 'Tensorflow' and `Tensorflow`.
 You can use pretrained model or the model trained by youself.

- [Mobilenet v1]()  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [Mobilenet v2]()  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [mobilenet-ssd]()  *caffe model can be found [here->](https://github.com/chuanqi305/MobileNet-SSD)*

## Test Results
The detailed test results can be seen here.
- [GPU](./README_GPU.md)
- [CPU](./README_CPU.md)
- [ARM](./README_ARM.md) 
