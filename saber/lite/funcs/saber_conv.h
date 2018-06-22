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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_CONV_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_CONV_H

#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/neon/impl/sgemm_arm.h"

namespace anakin{

namespace saber{

namespace lite{

typedef void (*conv_func)(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const float* weights, const float* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Sgemm& gemmer, void* work_space);


//template <typename Dtype>
class SaberConv2D {
public:
    SaberConv2D();

    SaberConv2D(int weights_size, int num_output, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, \
        bool flag_bias, const float* weights, const float* bias);

    SaberStatus load_param(int weights_size, int num_output, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, \
        bool flag_bias, const float* weights, const float* bias);

    ~SaberConv2D() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs);

    SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx);

    SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT>*>& outputs);

    SaberStatus set_activation(bool flag);

private:
    Context _ctx;
    conv_func _impl{nullptr};
    Sgemm _gemmer;
    bool _flag_relu{false};
    bool _is_trans_weights{false};
    bool _bias_term{true};
    int _num_output;
    int _group;
    int _kw;
    int _kh;
    int _stride_w;
    int _stride_h;
    int _pad_w;
    int _pad_h;
    int _dila_w;
    int _dila_h;
    const float* _weights{nullptr};
    const float* _bias{nullptr};
    int _weights_size;
    size_t _workspace_fwd_sizes{0};
    Tensor<CPU, AK_FLOAT> _workspace_data;
    Tensor<CPU, AK_FLOAT> _weights_trans;
};


} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_H
