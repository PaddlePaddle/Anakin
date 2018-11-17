/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SASS_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SASS_DECONV_H

#include <memory>
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype = AK_FLOAT>
class SassDeconv : public ImplBase<
        TargetType, OpDtype, ConvParam <TargetType>> {
public:
    typedef typename DataTrait<TargetType, OpDtype>::Dtype OpDataType;

    SassDeconv() = default;
    ~SassDeconv() = default;

    virtual SaberStatus init(const std::vector<Tensor<TargetType> *>& inputs,
                             std::vector<Tensor<TargetType>*>& outputs,
                             ConvParam<TargetType> &param, Context<TargetType>&ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<TargetType> *>& inputs,
                               std::vector<Tensor<TargetType>*>& outputs,
                               ConvParam<TargetType> &param,
                               Context<TargetType>&ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<TargetType> *>& inputs,
                                 std::vector<Tensor < TargetType>*>& outputs,
                                 ConvParam <TargetType> &param) override;

    SaberStatus trans_weights(Tensor<TargetType> &target_weights,
                              Tensor<TargetType> &target_bias,
                              int in_channel, int out_channel,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int dilation_h, int dilation_w,
                              int group) {

        if (target_weights.valid_size() > 0) {
            scale_to_new_tensor_k4_s2_p1_deconv<Tensor<NV>, Tensor<NVHX86>, 4>(
                    target_weights, in_channel, out_channel, true, nullptr);
        }
        _in_place = true;
        _extern_trans = true;
        return SaberSuccess;
    }

private:
    bool _with_saber_act{false};
    bool _in_place{false};
    bool _extern_trans{false};
    Tensor<TargetType> _weight_dev;
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_CUDA_SASS_DECONV_H
