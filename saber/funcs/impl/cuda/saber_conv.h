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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV2D_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV2D_H

#include <vector>
#include "saber/funcs/impl/impl_conv.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
#include "saber/funcs/impl/cuda/saber_activation.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin{

namespace saber{

template <typename dtype, bool bias_flag, bool relu_flag>
SaberStatus saber_depthwise_conv_act(const dtype* input, dtype* output, \
    int num, int cin, int hin, int win, int hout, int wout, \
    int kw, int kh, int stride_w, int stride_h, \
    int pad_h, int pad_w, const dtype* weights, const dtype* bias, \
    cudaStream_t stream);

template <DataType OpDtype>
class SaberConv2D<NV, OpDtype> : public ImplBase<
        NV, OpDtype, ConvParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberConv2D() = default;
    ~SaberConv2D() {
        delete _saber_act;
    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             ConvParam<NV>& param, Context<NV> &ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               ConvParam<NV>& param, Context<NV>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 ConvParam<NV>& param);

    void trans_weights(const std::vector<Tensor<NV> *>& inputs,
                       std::vector<Tensor<NV> *>& outputs,
                       ConvParam<NV>& param, Context<NV> &ctx) {

        Tensor<X86> trans_weights_host;
        if (param.stride_h == 1 &&
            param.stride_w == 1 &&
            param.weight()->height() == 3 &&
            param.weight()->width() == 3 && param.group == 1) {
            //Update weights if need
            Shape weight_shape = param.weight()->shape();
            Tensor<X86> new_weight;
            new_weight.re_alloc(weight_shape, param.weight()->get_dtype());
            new_weight.copy_from(*(param.weight()));
            OpDataType *weight_data = new_weight.mutable_data();
            int round_in_channel = i_align_up(inputs[0]->channel(), 8);
            int round_out_channel = i_align_up(param.weight()->num(), 32);
            int weight4x4_size = round_in_channel * round_out_channel * 4 * 4;
            Shape old_shape = param.weight()->shape();
            Shape new_trans_weights_shape({{weight4x4_size, 1, 1 ,1}}, param.weight()->get_layout());
            trans_weights_host.re_alloc(new_trans_weights_shape, param.weight()->get_dtype());
            OpDataType* _host_work_space = trans_weights_host.mutable_data();
            transform_3x3_weight_2_4x4(weight_data, _host_work_space, param.weight()->num(),
                                       round_out_channel, inputs[0]->channel(), round_in_channel);
            Shape new_weights_shape({weight4x4_size, 1, 1, 1}, param.weight()->get_layout());
            param.mutable_weight()->re_alloc(new_weights_shape, param.weight()->get_dtype());
            param.mutable_weight()->copy_from(trans_weights_host);
            param.mutable_weight()->set_shape(old_shape);

        } else if (param.group == 1) {
            int weight_size = (param.weight()->shape()).count();
            Tensor<X86> weight_host;
            weight_host.re_alloc(param.weight()->shape(), param.weight()->get_dtype());
            weight_host.copy_from(*(param.weight()));
            const OpDataType *weight_data = weight_host.data();
            trans_weights_host.re_alloc(param.weight()->valid_shape(), param.weight()->get_dtype());
            OpDataType* _host_work_space = trans_weights_host.mutable_data();

            transpose_filter_KCRS_2_CRSK(weight_data, _host_work_space, \
                                         param.weight()->num(), \
                                         param.weight()->channel(), \
                                         param.weight()->height(), \
                                         param.weight()->width());

            param.mutable_weight()->re_alloc(param.weight()->valid_shape(), param.weight()->get_dtype());
            param.mutable_weight()->copy_from(trans_weights_host);
        }
        cudaDeviceSynchronize();
    }

private:
    bool _with_saber_act{false};
    SaberActivation<NV, OpDtype> *_saber_act{nullptr};
    int _kernel_height;
    int _kernel_width;
    std::function<void(const float*,
                       float*,
                       const OpDataType*,
                       const float*,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       float,
                       float,
                       cudaStream_t)> dispatch_func;

    std::function<void(const float*, float* ,
                       int, int, int, int, int, int,
                       int, int, int, int,
                       int, int, const float*, const float*,
                       cudaStream_t)> depthwise_func;
};
}

}


#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
