/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#ifndef ANAKIN_OPERATOR_RPN_PROPOSAL_SSD_H
#define ANAKIN_OPERATOR_RPN_PROPOSAL_SSD_H
#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/rpn_proposal_ssd.h"
namespace anakin {
namespace ops {
template<typename Ttype, Precision Ptype>
class RPNProposalSSDHelper;
/// pooling op
/**
 * \brief RPNProposalSSD implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class RPNProposalSSD : public Operator<Ttype, Ptype> {
public:
    RPNProposalSSD() {}
    /// forward impl
    virtual void operator()(OpContext<Ttype>& ctx,
                            const std::vector<Tensor4dPtr<Ttype> >& ins,
                            std::vector<Tensor4dPtr<Ttype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator convolution<TargetType:" << "unknown" << ",";
    }
    friend class RPNProposalSSDHelper<Ttype, Ptype>;
};
/**
* \brief RPNProposalSSD helper class to implement it
* public inherit OperatorHelper
* including init resource and shape size in RPNProposalSSD context
*/
template<typename Ttype, Precision Ptype>
class RPNProposalSSDHelper : public OperatorHelper<Ttype, Ptype> {
public:
    RPNProposalSSDHelper() = default;
    ~RPNProposalSSDHelper();
    Status InitParam() override;
    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for RPNProposalSSD operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype>& ctx,
                const std::vector<Tensor4dPtr<Ttype> >& ins,
                std::vector<Tensor4dPtr<Ttype> >& outs) override;
    /**
    * \brief infer the shape of output and input.
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                      std::vector<Tensor4dPtr<Ttype> >& outs) override;
public:
    ///< _param_rpn_prop_ssd stand for RPNProposalSSD parameter
    saber::ProposalParam<Ttype>  _param_rpn_prop_ssd;
    ///< _funcs_rpn_prop_ssd stand for RPNProposalSSD function
    saber::RPNProposalSSD<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_rpn_prop_ssd;
};
} /* namespace ops */
} /* namespace anakin */
#endif