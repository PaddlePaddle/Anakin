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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_GEMMLIKE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_GEMMLIKE_H

#include <vector>
#include "saber/funcs/impl/impl_conv.h"
#include "sass_funcs.h"
#include "saber/funcs/impl/cuda/saber_activation.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberGemmLikeConv : public ImplBase<
        NV, OpDtype, ConvParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    typedef ImplBase<NV, OpDtype, ConvParam<NV> > Impl_t;
    SaberGemmLikeConv() = default;
    ~SaberGemmLikeConv() {
        delete _saber_act;
    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             ConvParam<NV>& param, Context<NV> &ctx) {
        this->_ctx = &ctx;
        if (param.activation_param.has_active) {
            if (param.activation_param.active != Active_relu) {
                _saber_act = new SaberActivation<NV, OpDtype>;
                _saber_act->init(inputs, outputs, param.activation_param, ctx);
            }
        }
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               ConvParam<NV>& param, Context<NV>& ctx) {
        if (_saber_act != nullptr) {
            _saber_act->create(outputs, outputs, param.activation_param, ctx);
        }
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 ConvParam<NV>& param);

private:
    SaberActivation<NV, OpDtype> *_saber_act{nullptr};

};
}

}


#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_GEMMLIKE_H
