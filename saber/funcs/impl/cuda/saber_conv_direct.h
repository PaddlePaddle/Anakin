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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_DIRECT_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_DIRECT_H

#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/cuda/saber_activation.h"
#include "saber/funcs/funcs_utils.h"
#include "sass_funcs.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberDirectConv : public ImplBase<
        NV, OpDtype, ConvParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    typedef ImplBase<NV, OpDtype, ConvParam<NV> > Impl_t;
    SaberDirectConv() = default;
    ~SaberDirectConv() {
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

private:
    bool _use_saber_act{false};
    SaberActivation<NV, OpDtype> *_saber_act{nullptr};
    //we use this func holder only when input and output datatype is float;
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
};
}

}


#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_DIRECT_H
