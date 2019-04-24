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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SPROPOSAL_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SPROPOSAL_H

#include "saber/funcs/impl/impl_sproposal.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSProposal<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        SProposalParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberSProposal()
    {}

    ~SaberSProposal() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            SProposalParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            SProposalParam<ARM>& param, Context<ARM> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          SProposalParam<ARM>& param);
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
    int _anchors_nums{0};
    int _src_height{0};
    int _src_width{0};
    float _src_scale{0};
    int _map_width{0};
    int _map_height{0};

    Tensor<ARM> _local_anchors;
    Tensor<ARM> _shift_x_tensor;
    Tensor<ARM> _shift_y_tensor;
    Tensor<ARM> _map_m_tensor;
    Tensor<ARM> _anchors_tensor;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Sproposal_H
