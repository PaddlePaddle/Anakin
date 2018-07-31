/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef ANAKIN_TEST_SABER_X86_TEST_CONV_COMMON_UTIL_H
#define ANAKIN_TEST_SABER_X86_TEST_CONV_COMMON_UTIL_H


template<DataType Dtype, typename LayoutType>
void compute_ref_conv_relu_fwd(
        const std::vector<Tensor<X86, Dtype, LayoutType> *> inputs,
        std::vector<Tensor<X86, Dtype, LayoutType> *> outputs,
        ConvParam<Tensor<X86, Dtype, LayoutType>> *conv_param,
        ActivationParam<Tensor<X86, Dtype, LayoutType>> *act_param){

    typedef typename Tensor<X86, Dtype, LayoutType>::Dtype dtype;  
    
    auto src_data = reinterpret_cast<const dtype*>(inputs[0]->get_buf()-> get_data());
    auto dst_data_ref = reinterpret_cast<dtype*>(outputs[0]->mutable_data());
    auto weights_data = reinterpret_cast<const dtype*>(conv_param->weight()->get_buf()->get_data());
    bool with_bias = conv_param->bias() ? true : false;
    auto bias_data = reinterpret_cast<const dtype*>(conv_param -> bias() -> data());
    Shape shape = conv_param->bias()->shape();
    int mb_ = outputs[0] -> num();
    int oc_ = outputs[0] -> channel();
    int oh_ = outputs[0] -> height();
    int ow_ = outputs[0] -> width();

    int ic_ = inputs[0] -> channel();
    int ih_ = inputs[0] -> height();
    int iw_ = inputs[0] -> width();

    int kh_ = conv_param -> weight() -> height();
    int kw_ = conv_param -> weight() -> width();
    int strh_ = conv_param -> stride_h;
    int strw_ = conv_param -> stride_w;
    int padh_ = conv_param -> pad_h;
    int padw_ = conv_param -> pad_w;
    int dilw_ = conv_param -> dilation_h;
    int dilh_ = conv_param -> dilation_w;

    dtype negative_slope = act_param -> negative_slope;

//#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < mb_; ++n) {
        for (int oc = 0; oc < oc_; ++oc) {
            for (int oh = 0; oh < oh_; ++oh) {
                for (int ow = 0; ow < ow_; ++ow) {
                    int oidx = n * oc_ * oh_ * ow_ 
                        + oc * oh_ * ow_ + oh * ow_ + ow;
       
                    dst_data_ref[oidx] = with_bias ? static_cast<dtype>(bias_data[oc]) : static_cast<dtype>(0);

                     for (int ic = 0; ic < ic_; ++ic){
                         for (int kh = 0; kh < kh_; ++kh) {
                             for (int kw = 0; kw < kw_; ++kw) {
                                 int iw = ow * strw_ - padw_ + kw * ( dilw_);
                                 int ih = oh * strh_ - padh_ + kh * ( dilh_);
                                 if (iw < 0 || iw >= iw_) continue;
                                 if (ih < 0 || ih >= ih_) continue;

                                 int iidx = n * ic_ * ih_ * iw_ +
                                     ic * ih_ * iw_ + ih * iw_ + iw;
                                 int widx = oc * ic_ * kh_ * kw_ +
                                     ic * kh_ * kw_ + kh * kw_ + kw;

                                 dst_data_ref[oidx] 
                                     += src_data[iidx] 
                                     * weights_data[widx];
                             }
                         }
                     }

                     if (dst_data_ref[oidx] < 0){
                         dst_data_ref[oidx] = static_cast<dtype>(
                                 negative_slope * dst_data_ref[oidx]);
                     }
                }
            }
        }
    }
}



#endif //ANAKIN_TEST_SABER_X86_TEST_CONV_COMMON_UTIL_H