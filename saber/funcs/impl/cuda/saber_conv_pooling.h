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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_POOLING_H

#include <vector>
#include "saber/funcs/impl/impl_conv_pooling.h"
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/impl/cuda/vender_conv.h"
#include "saber/funcs/impl/cuda/vender_pooling.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin {

namespace saber {

template<DataType OpDtype>
class SaberConv2DPooling<NV, OpDtype> : public ImplBase<
        NV, OpDtype, ConvPoolingParam<NV>> {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberConv2DPooling() = default;
    ~SaberConv2DPooling() = default;

    virtual SaberStatus init(const std::vector<Tensor<NV> *> &inputs,
                             std::vector<Tensor<NV> *> &outputs,
                             ConvPoolingParam<NV> &param, Context<NV>
                             &ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *> &inputs,
                               std::vector<Tensor<NV> *> &outputs,
                               ConvPoolingParam<NV> &param, Context<NV>
                               &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *> &inputs,
                                 std::vector<Tensor<NV> *> &outputs,
                                 ConvPoolingParam<NV> &param);

    SaberStatus trans_weights(const std::vector<Tensor<NV> *>& inputs,
                              std::vector<Tensor<NV> *>& outputs,
                              ConvParam<NV>& param, Context<NV> &ctx,
                              bool in_place = false, Tensor<NV>* weight_dev = nullptr) {
        conv_trans_weights<NV, NVHX86>(inputs, outputs, param, ctx, in_place, weight_dev);
        return SaberSuccess;
    }
private:
    bool _use_k3p{false};
    bool _use_kp{false};
    bool _in_place{false};
    Tensor<NV> _weight_dev;
    VenderPooling<NV, OpDtype> _vender_pool;
    SaberConv2D<NV, OpDtype> _saber_conv;
    Shape _inner_shape;
    Tensor<NV> _inner_tensor;
    std::vector<Tensor<NV> *> _inner_tensor_v;
    int _kernel_height;
    int _kernel_width;

    std::function<void(const float*,
                       float*,
                       const float*,
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
};
}

}


#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
