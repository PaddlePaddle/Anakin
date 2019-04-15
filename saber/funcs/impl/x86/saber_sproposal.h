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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SPROPOSAL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SPROPOSAL_H

#include "saber/funcs/impl/impl_sproposal.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSProposal<X86, OpDtype>:
        public ImplBase<X86, OpDtype, SProposalParam<X86>> {

public:

    SaberSProposal() = default;

    ~SaberSProposal() = default;

    SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            SProposalParam<X86> &param,
            Context<X86> &ctx) override;

    SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            SProposalParam<X86> &param,
            Context<X86> &ctx) override;

    SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            SProposalParam<X86> &param) override;

private:
    void generate_anchors();
    std::vector<std::vector<float> > ratio_enum(std::vector<float>);
    std::vector<float> whctrs(std::vector<float>);
    std::vector<float> mkanchor(float w,float h,float x_ctr,float y_ctr);
    std::vector<std::vector<float> > scale_enum(std::vector<float>);

    int _feat_stride{0};
    int _base_size{0};
    int _min_size{0};
    int _pre_nms_topN{0};
    int _post_nms_topN{0};
    float _nms_thresh{0};
    std::vector<int> _anchor_scales;
    std::vector<float> _ratios;

    std::vector<std::vector<float> > _gen_anchors;
    int *_anchors{nullptr};
    int _anchors_nums{0};
    int _src_height{0};
    int _src_width{0};
    float _src_scale{0};
    int _map_width{0};
    int _map_height{0};

    Tensor<X86> _local_anchors;
    Tensor<X86> _shift_x_tensor;
    Tensor<X86> _shift_y_tensor;
    Tensor<X86> _map_m_tensor;
    Tensor<X86> _anchors_tensor;
};

}

}

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SPROPOSAL_H
