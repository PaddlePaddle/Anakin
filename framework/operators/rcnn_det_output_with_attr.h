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

#ifndef ANAKIN_OPERATOR_RCNN_DET_OUTPUT_WITH_ATTR_H
#define ANAKIN_OPERATOR_RCNN_DET_OUTPUT_WITH_ATTR_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/rcnn_det_output_with_attr.h"

namespace anakin {
namespace ops {
template<typename Ttype, Precision Ptype>
class RCNNDetOutputWithAttrHelper;
/// pooling op
/**
 * \brief RCNNDetOutputWithAttr implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class RCNNDetOutputWithAttr : public Operator<Ttype, Ptype> {
public:
    RCNNDetOutputWithAttr() {}
    /// forward impl
    virtual void operator()(OpContext<Ttype>& ctx,
                            const std::vector<Tensor4dPtr<Ttype> >& ins,
                            std::vector<Tensor4dPtr<Ttype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator RCNNDetOutputWithAttr<TargetType:" << "unknown" << ","
                    << ">";
    }
    friend class RCNNDetOutputWithAttrHelper<Ttype, Ptype>;
};
/**
 * \brief RCNNDetOutputWithAttr helper class to implement it
 * public inherit OperatorHelper
 * including init resource and shape size in RCNNDetOutputWithAttr context
 */
template<typename Ttype, Precision Ptype>
class RCNNDetOutputWithAttrHelper : public OperatorHelper<Ttype, Ptype> {
public:
    RCNNDetOutputWithAttrHelper() = default;
    ~RCNNDetOutputWithAttrHelper();
    Status InitParam() override;
    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for RCNNDetOutputWithAttr operation context
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
    ///< _param_rpn_prop_ssd stand for RCNNDetOutputWithAttr parameter
    saber::ProposalParam<Ttype>  _param_rpn_prop_ssd;
    ///< _funcs_conv stand for convolution function
    saber::RCNNDetOutputWithAttr<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_rpn_prop_ssd;
};
} /* namespace ops */
} /* namespace anakin */
#endif