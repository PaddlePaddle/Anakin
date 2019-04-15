/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_BOX_CLIP_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_BOX_CLIP_H

#include "anakin_config.h"
#include "saber/funcs/impl/impl_box_clip.h"
#include "saber/core/tensor.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberBoxClip<NV, OpDtype> : \
    public ImplBase <
    NV,
    OpDtype,
    EmptyParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberBoxClip() = default;
    ~SaberBoxClip() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             EmptyParam<NV>& param, Context<NV>& ctx) {
        // get context
        this->_ctx = &ctx;
        cuda_seq_offset.re_alloc(Shape({1, 1, 1, 1}), AK_FLOAT);
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               EmptyParam<NV>& param, Context<NV>& ctx) {

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 EmptyParam<NV>& param)override;

private:
    Tensor<NV> cuda_seq_offset;
};

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_BOX_CLIP_H
