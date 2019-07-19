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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SROI_ALIGN_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SROI_ALIGN_H

#include "saber/funcs/impl/impl_sroi_align.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSRoiAlign<X86, OpDtype>:
        public ImplBase<X86, OpDtype, SRoiAlignParam<X86>> {

public:

    SaberSRoiAlign() = default;

    ~SaberSRoiAlign() = default;

    SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            SRoiAlignParam<X86> &param,
            Context<X86> &ctx) override;

    SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            SRoiAlignParam<X86> &param,
            Context<X86> &ctx) override;

    SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            SRoiAlignParam<X86> &param) override;

private:
    int _channels;
    int _height;
    int _width;
    int _pooled_height;
    int _pooled_width;
    float _spatial_scale;
};

}

}

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SROI_ALIGN_H