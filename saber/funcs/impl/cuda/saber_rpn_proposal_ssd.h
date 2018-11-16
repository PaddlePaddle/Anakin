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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_RPN_PROPOSAL_SSD_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_RPN_PROPOSAL_SSD_H

#include "anakin_config.h"
#include <vector>
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include "saber/funcs/impl/impl_rpn_proposal_ssd.h"
#include "saber/utils.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberRPNProposalSSD<NV, OpDtype> : public ImplROIOutputSSD <
        NV, OpDtype > {

public:

    SaberRPNProposalSSD()
            : box_dev_nms_(NULL)
            , boxes_dev_len(0)
            , mask_dev_nms_(NULL)
    {}

    ~SaberRPNProposalSSD() {
        if (box_dev_nms_ != NULL) {
            CUDA_CHECK(cudaFree(box_dev_nms_));
        }
        if (mask_dev_nms_ != NULL) {
            CUDA_CHECK(cudaFree(mask_dev_nms_));
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<NV>*> &inputs,
                             std::vector<Tensor<NV>*> &outputs,
                             ProposalParam<NV> &param, Context<NV> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<NV>*> &inputs,
                               std::vector<Tensor<NV>*> &outputs,
                               ProposalParam<NV> &param, Context<NV> &ctx) override;

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*> &inputs,
                                 std::vector<Tensor<NV>*> &outputs,
                                 ProposalParam<NV> &param);

private:
    int num_rpns_;
    int num_anchors_;
    bool has_img_info_;
    int rois_dim_;

    // ADD CPU TENSORS
    PGlue<Tensor<NV>, Tensor<NVHX86> > _img_info_glue;
    PGlue<Tensor<NV>, Tensor<NVHX86> > _outputs_boxes_scores_glue;
    PGlue<Tensor<NV>, Tensor<NVHX86> > anc_;
    PGlue<Tensor<NV>, Tensor<NVHX86> > dt_conf_ahw_;
    PGlue<Tensor<NV>, Tensor<NVHX86> > dt_bbox_ahw_;
    PGlue<Tensor<NV>, Tensor<NVHX86> > overlapped_;
    PGlue<Tensor<NV>, Tensor<NVHX86> > idx_sm_;

    //caffe pyramid_layers.hpp:615
    float* box_dev_nms_;
    unsigned long long* mask_dev_nms_;
    int boxes_dev_len;
    //caffe pyramid_layers.hpp:618
};

}

}

#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H