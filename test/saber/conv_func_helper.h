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

#include "saber/core/tensor.h"

namespace anakin {
namespace saber {

template<typename targetType>
void conv_basic_check(Tensor<targetType> &tensor_out, Tensor<targetType> &tensor_in,
                      const float *weights, const float *bias, int group,
                      int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h,
                      int pad_w, int pad_h, bool flag_bias, bool flag_relu) {

    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num_in = tensor_in.num();

    int w_out = tensor_out.width();
    int h_out = tensor_out.height();
    int ch_out = tensor_out.channel();

    const int size_kernel = kernel_h * kernel_w;

    int kernel_ext_w = (kernel_w - 1) * dila_w + 1;
    int kernel_ext_h = (kernel_h - 1) * dila_h + 1;

    const int ch_out_g = ch_out / group;
    const int ch_in_g = ch_in / group;
    const int size_in_channel = w_in * h_in;
    const int size_in_batch = size_in_channel * ch_in;
    const int size_out_channel = w_out * h_out;
    const int size_out_batch = size_out_channel * ch_out;
    const float *data_in = tensor_in.data();
    float *outptr = tensor_out.mutable_data();

    for (int b = 0; b < num_in; ++b) {
        float *outptr_batch = outptr + b * size_out_batch;
        const float *data_in_batch = data_in + b * size_in_batch;
        for (int g = 0; g < group; ++g) {
            for (int c = 0; c < ch_out_g; ++c) {
                const float *inptr_group = data_in_batch + g * ch_in_g * size_in_channel;
                float *outptr_ch = outptr_batch + (g * ch_out_g + c) * size_out_channel;
                const float *weight_ch = weights + (g * ch_out_g + c) * ch_in_g * size_kernel;

                float bias_value = flag_bias ? bias[g * ch_out_g + c] : 0.f;
//                fill_bias(outptr_ch, &bias_value, 1, w_out * h_out);

                for (int i = 0; i < h_out; ++i) {
                    for (int j = 0; j < w_out; ++j) {

                        const float *weight_ch_in = weight_ch;

                        int hstart = i * stride_h - pad_h;
                        int wstart = j * stride_w - pad_w;
                        int hend = std::min(hstart + kernel_ext_h, h_in);
                        int wend = std::min(wstart + kernel_ext_w, w_in);
                        hstart = std::max(hstart, 0);
                        wstart = std::max(wstart, 0);

                        int khstart = hend < kernel_ext_h ? (kernel_ext_h - hend) / dila_h : 0;
                        int kwstart = wend < kernel_ext_w ? (kernel_ext_w - wend) / dila_w : 0;

                        const float *inptr_ch = inptr_group + hstart * w_in + wstart;

                        for (int k = 0; k < ch_in_g; ++k) {
                            const float *inptr_kernel = inptr_ch;
                            int khidx = khstart;
                            for (int idxh = hstart; idxh < hend; idxh += dila_h, khidx++) {
                                const float *inptr_kernel_w = inptr_kernel;
                                int kwidx = kwstart;
                                for (int idxw = wstart; idxw < wend; idxw += dila_w, kwidx++) {
                                    outptr_ch[j] += weight_ch_in[khidx * kernel_w + kwidx] * inptr_kernel_w[0];
                                    inptr_kernel_w += dila_w;
                                }
                                inptr_kernel += dila_h * w_in;
                            }
                            inptr_ch += size_in_channel;
                            weight_ch_in += size_kernel;
                        }
                        if (flag_bias) {
                            outptr_ch[j] += bias_value;
                        }
                        if (flag_relu) {
                            outptr_ch[j] = outptr_ch[j] > 0 ? outptr_ch[j] : 0.f;
                        }
                    }
                    outptr_ch += w_out;
                }
            }
        }
    }
}
}
}
#endif //ANAKIN_CONV_FUNC_HELPER_H
