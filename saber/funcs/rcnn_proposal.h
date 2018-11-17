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
#ifndef ANAKIN_SABER_FUNCS_RCNN_PROPOSAL_H
#define ANAKIN_SABER_FUNCS_RCNN_PROPOSAL_H

#include "saber/funcs/base.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_rcnn_proposal.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_rcnn_proposal.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_rcnn_proposal.h"
#endif

namespace anakin {
namespace saber {
template <typename TargetType,
        DataType OpDtype>
class RCNNProposal : public BaseFunc <
        TargetType, OpDtype,
        ImplBase, ProposalParam
> {
public:
    typedef TargetType targetType_t;
    typedef Tensor<TargetType> OpTensor;
    typedef ProposalParam<TargetType> Param_t;
    typedef const std::vector<OpTensor*> Input_v;
    typedef std::vector<OpTensor*> Output_v;

    RCNNProposal()= default;
    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) {
        int rois_dim = param.detection_output_ssd_param.rpn_proposal_output_score ? (
                5 + param.detection_output_ssd_param.num_class + 1) : 5;
        Shape dummy_output_shape({300, rois_dim, 1, 1}, Layout_NCHW);

        for (int i = 0; i < output.size(); ++i) {
            output[i]->set_shape(dummy_output_shape);
        }
        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderRCNNProposal<TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberRCNNProposal<TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }

    }

private:
    virtual void pick_best_static() override {
        if (true) { // some condition?
            this->_best_impl = this->_impl[0];
        }
    }
    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }
};
}
}
#endif //ANAKIN_SABER_FUNCS_CONV_H
