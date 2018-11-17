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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H

#include "saber/funcs/impl/impl_deconv.h"
#include "sass_funcs.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberDeconv2D<NV, OpDtype> :
        public ImplBase<NV, OpDtype, ConvParam<NV> > {
public:
    typedef ImplBase<NV, OpDtype, ConvParam<NV> > Impl_t;
    SaberDeconv2D() = default;

    ~SaberDeconv2D() {
        if (_impl) {
            delete _impl;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             ConvParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               ConvParam<NV>& param, Context<NV> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 ConvParam<NV>& param);

    SaberStatus trans_weights(Tensor<NV> &target_weights,
                              Tensor<NV> &target_bias,
                              int in_channel, int out_channel,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int dilation_h, int dilation_w,
                              int group);
private:
    bool _use_k4_s2_p1{false};

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wx;

    Impl_t* _impl{nullptr};
};

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H
