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

#include "saber/funcs/impl/impl_conv.h"
#include "saber/funcs/impl/cuda/saber_activation.h"
#include "saber/funcs/funcs_utils.h"
#include "sass_funcs.h"
#include <vector>

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
                             ConvParam<NV>& param, Context<NV> &ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               ConvParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 ConvParam<NV>& param);

    void set_act(bool use_act) {
        _use_act = use_act;
    }
private:
    SaberActivation<NV, OpDtype> *_saber_act{nullptr};
    bool _use_act{false};
    std::function<void(int, int, int, void *, const void *, const void *,
            int, int, int, int, const void *, cudaStream_t,
            float, float, int)> _int8_func;

};
}

}


#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_GEMMLIKE_H
