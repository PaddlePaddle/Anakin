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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_RPN_PROPOSAL_SSD_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_RPN_PROPOSAL_SSD_H

#include "anakin_config.h"
#include <vector>
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include "saber/funcs/impl/impl_rpn_proposal_ssd.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberRPNProposalSSD<X86, OpDtype> : public ImplROIOutputSSD <
        X86, OpDtype > {

public:

    SaberRPNProposalSSD() = default;

    ~SaberRPNProposalSSD() {
        if (_img_info_data_host_tensor != NULL) {
            delete _img_info_data_host_tensor;
        }
        if (_prob_data_host_tensor != NULL) {
            delete _prob_data_host_tensor;
        }
        if (_tgt_data_host_tensor != NULL) {
            delete _tgt_data_host_tensor;
        }
        if (_outputs_boxes_scores_host_tensor != NULL) {
            delete _outputs_boxes_scores_host_tensor;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<X86>*> &inputs,
                             std::vector<Tensor<X86>*> &outputs,
                             ProposalParam<X86> &param, Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86>*> &inputs,
                               std::vector<Tensor<X86>*> &outputs,
                               ProposalParam<X86> &param, Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*> &inputs,
                                 std::vector<Tensor<X86>*> &outputs,
                                 ProposalParam<X86> &param) override;

private:
    int num_rpns_{0};
    int num_anchors_{0};
    bool has_img_info_{false};
    int rois_dim_{0};

    // ADD CPU TENSORS
    Tensor<X86> *_img_info_data_host_tensor{nullptr};
    Tensor<X86> *_prob_data_host_tensor{nullptr};
    Tensor<X86> *_tgt_data_host_tensor{nullptr};
    Tensor<X86> *_outputs_boxes_scores_host_tensor{nullptr};

    //caffe pyramid_layers.hpp:615
    float* box_dev_nms_{nullptr};
    unsigned long long* mask_dev_nms_{nullptr};
    int boxes_dev_len{0};
    //caffe pyramid_layers.hpp:618
};

}

}

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_RPN_PROPOSAL_SSD_H