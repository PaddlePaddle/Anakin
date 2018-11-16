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
#ifndef ANAKIN_OPERATOR_RCNN_PROPOSAL_H
#define ANAKIN_OPERATOR_RCNN_PROPOSAL_H
#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/rcnn_proposal.h"
namespace anakin {
namespace ops {
template<typename Ttype, Precision Ptype>
class RCNNProposalHelper;
/// pooling op
/**
 * \brief RCNNProposal implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class RCNNProposal : public Operator<Ttype, Ptype> {
public:
    RCNNProposal() {}
    /// forward impl
    virtual void operator()(OpContext<Ttype>& ctx,
                            const std::vector<Tensor4dPtr<Ttype>>& ins,
                            std::vector<Tensor4dPtr<Ttype>>& outs) {
        LOG(ERROR) << "Not Impl Yet Operator convolution<TargetType:" << "unknown" << "," << ">";
    }
    friend class RCNNProposalHelper<Ttype, Ptype>;
};
/**
 * \brief RCNNProposal helper class to implement RCNNProposal
 * public inherit OperatorHelper
 * including init resource and shape size in RCNNProposal context
 */
template<typename Ttype, Precision Ptype>
class RCNNProposalHelper : public OperatorHelper<Ttype, Ptype> {
public:
    RCNNProposalHelper() = default;
    ~RCNNProposalHelper();
    Status InitParam() override;
    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for RCNNProposal operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    *///! initial all the resource needed by pooling
    Status Init(OpContext<Ttype>& ctx,
                const std::vector<Tensor4dPtr<Ttype>>& ins,
                std::vector<Tensor4dPtr<Ttype>>& outs) override;
    /**
    * \brief infer the shape of output and input.
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype>>& ins,
                      std::vector<Tensor4dPtr<Ttype>>& outs) override;
public:
    ///< _param__rcnn_prop stand for RCNNProposal parameter
    saber::ProposalParam<Ttype>  _param__rcnn_prop;
    ///< _funcs_rcnn_prop stand for RCNNProposal function
    saber::RCNNProposal<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_rcnn_prop;
};
} /* namespace ops */
} /* namespace anakin */
#endif