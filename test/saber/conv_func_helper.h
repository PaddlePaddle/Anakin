/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef ANAKIN_CONV_FUNC_HELPER_H
#define ANAKIN_CONV_FUNC_HELPER_H

#include "saber/core/context.h"
#include "saber/core/tensor.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/conv.h"
#include "saber/saber_types.h"
#include <vector>

namespace anakin {
namespace saber {

template<typename targetType, typename out_dtype = float, typename in_dtype = out_dtype>
void conv_basic_check(Tensor<targetType> &tensor_in,Tensor<targetType> &tensor_out,
                      const in_dtype *weights, const out_dtype *bias, int group,
                      int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
                      int pad_w, int pad_h, bool flag_bias, bool flag_relu, float beta = 0.f) {

    auto src_data = reinterpret_cast<const in_dtype*>(tensor_in.data());
    auto dst_data_ref = reinterpret_cast<out_dtype*>(tensor_out.mutable_data());
    auto weights_data = weights;
    bool with_bias = flag_bias;
    auto bias_data = bias;

    int in_num = tensor_out.num();
    int out_channels = tensor_out.channel();
    int out_h = tensor_out.height();
    int out_w = tensor_out.width();

    int in_channel = tensor_in.channel();
    int in_h = tensor_in.height();
    int in_w = tensor_in.width();
    int out_c_group = out_channels / group;
    int in_c_group = in_channel / group;
#pragma omp parallel for num_threads(8) collapse(5) schedule(static)
    for (int n = 0; n < in_num; ++n) {
        for (int g = 0; g < group; ++g) {
            for (int oc = 0; oc < out_c_group; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        int out_idx = n * group * out_c_group * out_h * out_w + g * out_c_group * out_h * out_w
                                   + oc * out_h * out_w + oh * out_w + ow;
                        float bias_d = with_bias ? (float)(bias_data[g * out_c_group + oc]) : 0.f;
                        dst_data_ref[out_idx] = bias_d + dst_data_ref[out_idx] * beta;
                        for (int ic = 0; ic < in_c_group; ++ic) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int iw = ow * stride_w - pad_w + kw * (dilation_w);
                                    int ih = oh * stride_h - pad_h + kh * (dilation_h);
                                    if (iw < 0 || iw >= in_w) continue;
                                    if (ih < 0 || ih >= in_h) continue;

                                    int iidx = n * in_channel * in_h * in_w
                                               + g * in_c_group * in_h * in_w
                                               + ic * in_h * in_w
                                               + ih * in_w
                                               + iw;
                                    int widx = g * out_c_group * in_c_group * kernel_h * kernel_w
                                               + oc * in_c_group * kernel_h * kernel_w
                                               + ic * kernel_h * kernel_w
                                               + kh * kernel_w
                                               + kw;

                                    dst_data_ref[out_idx]
                                            += (out_dtype)src_data[iidx]
                                               * (out_dtype)weights_data[widx];
//                                            LOG(INFO) << "out_idx = " << out_idx << " iidx = " << iidx << " res = " << dst_data_ref[out_idx];
                                }
                            }
                        }
                        if (flag_relu) {
                            dst_data_ref[out_idx] = dst_data_ref[out_idx] > 0.f ? dst_data_ref[out_idx] : 0.f;
                        }
                    }
                }
            }
        }
    }
}

template<typename dtype,typename TargetType_D,typename TargetType_H>
void conv_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
                    std::vector<Tensor<TargetType_H>*>& output,
                    ConvParam<TargetType_D>& param) {
    int group = param.group;
    int input_num = input[0]->num();
    int input_channel = input[0]->channel();
    int input_height = input[0]->height();
    int input_width = input[0]->width();
    int output_channel = output[0]->channel();
    int output_height = output[0]->height();
    int output_width = output[0]->width();
    int stride_h = param.stride_h;
    int stride_w = param.stride_w;
    int dilation_h = param.dilation_h;
    int dilation_w = param.dilation_w;
    int pad_h = param.pad_h;
    int pad_w = param.pad_w;
    int kernel_h = param.weight()->height();
    int kernel_w = param.weight()->width();
    bool bias_term = param.bias()->valid_size() > 0;
    bool with_relu = param.activation_param.has_active;

    Tensor<TargetType_H> weights_host;
    Tensor<TargetType_H> bias_host;
    weights_host.re_alloc(param.weight()->valid_shape(), AK_FLOAT);
    weights_host.copy_from(*(param.weight()));
    bias_host.re_alloc(param.bias()->valid_shape(), AK_FLOAT);
    bias_host.copy_from(*(param.bias()));

    const dtype* bias_ptr = bias_term ? (const float*)bias_host.data() : nullptr;
    conv_basic_check<TargetType_H>(*input[0], *output[0],
    (const dtype*)weights_host.data(), bias_ptr,
            group, kernel_w, kernel_h, stride_w, stride_h,
            dilation_w, dilation_h, pad_w, pad_h, bias_term,
            with_relu);
}
}
}
#endif //ANAKIN_CONV_FUNC_HELPER_H
