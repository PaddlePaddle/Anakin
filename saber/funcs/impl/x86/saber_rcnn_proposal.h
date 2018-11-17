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
class SaberRCNNProposal<X86, OpDtype> : public ImplROIOutputSSD <
        X86, OpDtype > {
public:

    SaberRCNNProposal()
            : _img_info_data_host_tensor(NULL)
            , _probs_st_host_tensor(NULL)
            , _cords_st_host_tensor(NULL)
            , _rois_st_host_tensor(NULL)
            , _outputs_boxes_scores_host_tensor(NULL)
            , has_img_info_(false)
            , rois_dim_(0)
    {}

    ~SaberRCNNProposal() {
        if (_img_info_data_host_tensor != NULL) {
            delete _img_info_data_host_tensor;
        }

        if (_probs_st_host_tensor != NULL) {
            delete _probs_st_host_tensor;
        }

        if (_cords_st_host_tensor != NULL) {
            delete _cords_st_host_tensor;
        }

        if (_rois_st_host_tensor != NULL) {
            delete _rois_st_host_tensor;
        }

        if (_outputs_boxes_scores_host_tensor != NULL) {
            delete _outputs_boxes_scores_host_tensor;
        }
    }
    virtual SaberStatus init(const std::vector<Tensor<X86>*> &inputs,
                             std::vector<Tensor<X86>*> &outputs,
                             ProposalParam<X86>& param,
                             Context<X86>& ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86>*> &inputs,
                               std::vector<Tensor<X86>*> &outputs,
                               ProposalParam<X86>& param,
                               Context<X86>& ctx) override;

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*> &inputs,
                                 std::vector<Tensor<X86>*> &outputs,
                                 ProposalParam<X86>& param);
private:
    bool has_img_info_;
    int rois_dim_;
    Tensor<X86>* _img_info_data_host_tensor;
    Tensor<X86>* _probs_st_host_tensor;
    Tensor<X86>* _cords_st_host_tensor;
    Tensor<X86>* _rois_st_host_tensor;
    Tensor<X86>* _outputs_boxes_scores_host_tensor;
};
}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_RCNN_PROPOSAL_H
