- [Convolution](#convolution)

# Convolution

Defined in [Anakin/framework/operators/convolution.cpp](../../framework/operators/convolution.cpp)

* 介绍：
  * 作用：对输入的4维张量和kernel进行一个二维卷积操作，通过设置参数来支持各种类型的卷积，其中包括group convolution和dilation convolution等等。
  * 公式：
    $$
    Output = W * X + Bias
    $$
    * X: 输入值。
    * W: 权重，滤波器值。 
    * Bias: Bias值。
    * Output: 输出值。  
* 参数：

  * input(tensor): 输入4维张量，格式为NCHW。
  * filter_num(int): 滤波器个数，和输出通道数相同。
  * kernel_size(vecotr<int>): 滤波器大小，二维数据，shape为[kernel_size_h, kernel_size_w]。
  * padding(vector<int>): padding大小，二维数据，shape为[padding_x, padding_y]。
  * strides(vector<int>): 步长，二维数据，shape为[strides_x, strides_y]。
  * dilation_rate(int): 实现空洞卷积的膨胀大小，在不丢失语义信息的情况下增大感受野。
  * group(int): group convolution的组数，对通道进行分组卷积。
  * bias_term(bool): 卷积操作是否计算偏移量，如果为true，一定存在bias weights。
  * axis(int): 滤波器处理的维度信息，默认为1。
  * weights(tensor): 滤波器权重，格式为NCHW。

* 输出：
  
  * Output(tensor): 卷积后的tensor，格式为NCHW。 
