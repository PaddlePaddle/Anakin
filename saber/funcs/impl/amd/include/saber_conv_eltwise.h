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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_CONV_ELTWISE_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_CONV_ELTWISE_H

#include <vector>
#include "saber/funcs/impl/amd/include/saber_conv.h"
#include "saber/funcs/impl/amd/include/saber_eltwise.h"
#include "saber/funcs/impl/impl_conv_eltwise.h"
#include "saber_conv_eltwise.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberConvEltwise<AMD, OpDtype> : public ImplBase<AMD, OpDtype, ConvEltwiseParam<AMD>> {
public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;

    SaberConvEltwise() = default;
    ~SaberConvEltwise() {}

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         ConvEltwiseParam<AMD>& param,
         Context<AMD>& ctx);

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           ConvEltwiseParam<AMD>& param,
           Context<AMD>& ctx);

    virtual SaberStatus dispatch(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        ConvEltwiseParam<AMD>& param);

    SaberStatus trans_weights(Tensor<AMD>& target_weights,
                              Tensor<AMD>& bias_weights, int pad_h, int pad_w, int dilation_h, int dilation_w, int stride_h,
                              int stride_w, int group) {
        return SaberSuccess;
    }

private:
    SaberConv2D<AMD, OpDtype> _conv;
    SaberEltwise<AMD, OpDtype> _eltwise;
    Shape _inner_shape;
    Tensor<AMD> _inner_tensor;
    std::vector<Tensor<AMD>*> _inner_tensor_v;
};
} // namespace saber

} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_SABER_CONV_ELTWISE_H
