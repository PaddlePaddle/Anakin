/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_KER_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_KER_DECONV_H

#include "saber/core/tensor.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin {
namespace saber {

template<int k>
void scale_to_new_tensor_k4_s2_p1_decov (Tensor<NV, AK_FLOAT, NCHW> &new_weights_dev,
                                         const Tensor<NV, AK_FLOAT, NCHW> *weight,
                                         int in_channel, int out_channel) {
    Tensor<X86, AK_FLOAT, NCHW> new_weights_h;
    Tensor<X86, AK_FLOAT, NCHW> temp_weights;
    new_weights_dev.reshape(weight->valid_shape());
    new_weights_h.reshape(weight->valid_shape());
    temp_weights.reshape(weight->valid_shape());

    temp_weights.copy_from(*weight);
    int offset = in_channel * out_channel * k;
    float* trans_w = new_weights_h.mutable_data();
    scale_weight_deconv_w4x4<k, true>(trans_w + 0 * offset,
                                      trans_w + 1 * offset,
                                      trans_w + 2 * offset,
                                      trans_w + 3 * offset,
                                      temp_weights.data(),
                                      in_channel, out_channel);
    new_weights_dev.copy_from(new_weights_h);
}

void ker_deconv_implicit_gemm_k4_s2_p1_16x64(float* dout, const float *din,
                                             const float* weights, const float* bias,
                                             int num, int hin, int win, int hout, int wout,
                                             int ch_in, int ch_out, cudaStream_t &stream);

void ker_deconv_implicit_gemm_k4_s2_p1_32x32_relu(float* dout, const float *din,
                                             const float* weights, const float* bias,
                                             int num, int hin, int win, int hout, int wout,
                                             int ch_in, int ch_out, cudaStream_t &stream);
}

}

#endif