/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_RCNN_PROPOSAL_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_RCNN_PROPOSAL_H

#include <vector>
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_rcnn_proposal.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberRCNNProposal<NV, OpDtype> : public ImplROIOutputSSD <
        NV, OpDtype > {
public:

    SaberRCNNProposal()
            : has_img_info_(false)
            , rois_dim_(0)
    {}

    ~SaberRCNNProposal() {

    }
    virtual SaberStatus init(const std::vector<Tensor<NV>*> &inputs,
                             std::vector<Tensor<NV>*> &outputs,
                             ProposalParam<NV>& param,
                             Context<NV>& ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<NV>*> &inputs,
                               std::vector<Tensor<NV>*> &outputs,
                               ProposalParam<NV>& param,
                               Context<NV>& ctx) override;

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*> &inputs,
                                 std::vector<Tensor<NV>*> &outputs,
                                 ProposalParam<NV>& param);
private:
    bool has_img_info_;
    int rois_dim_;
    PGlue<Tensor<NV>, Tensor<NVHX86> > _img_info_glue;
    PGlue<Tensor<NV>, Tensor<NVHX86> > _probs_st_glue;
    PGlue<Tensor<NV>, Tensor<NVHX86> > _rois_st_glue;
    PGlue<Tensor<NV>, Tensor<NVHX86> > _outputs_boxes_scores_glue;
    PGlue<Tensor<NV>, Tensor<NVHX86> > dt_conf_;
    PGlue<Tensor<NV>, Tensor<NVHX86> > thr_cls_;
    PGlue<Tensor<NV>, Tensor<NVHX86> > dt_bbox_;
    PGlue<Tensor<NV>, Tensor<NVHX86> > overlapped_;
    PGlue<Tensor<NV>, Tensor<NVHX86> > idx_sm_;
};
}
}
#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
