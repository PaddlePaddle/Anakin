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
                               ConvParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 ConvParam<NV>& param);

    SaberStatus trans_weights(Tensor<NV> &target_weights,
            int stride_h, int stride_w, int group) {
        conv_trans_weights<NV, NVHX86>(target_weights, stride_h, stride_w, group, true, nullptr);
        _extern_trans = true;
        return SaberSuccess;
    }

private:
    bool _with_saber_act{false};
    bool _in_place{false};
    bool _use_k1s1p0{false};
    bool _extern_trans{false};
    Tensor<NV> _weight_dev;
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
