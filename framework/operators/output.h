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

#ifndef ANAKIN_OPERATOR_OUTPUT_H
#define ANAKIN_OPERATOR_OUTPUT_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class OutputHelper;

/// Output op without any compute, this a holder for input
/**
 * \brief Output implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Output : public Operator<Ttype, Dtype, Ptype> {
public:
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator Output<TargetType:"<<"unknown"<<"," 
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info();
    }

    friend class OutputHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_OUTPUT(Ttype, Dtype, Ptype) \
template<> \
void Output<Ttype, Dtype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype>>& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype>>& outs) {}
/**
 * \brief Output helper class
 * public inherit OperatorHelper
 * including init resource and shape size in output context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class OutputHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
    typedef OperatorHelper<Ttype, Dtype, Ptype> Base; 
public:
    OutputHelper() {}

    ~OutputHelper(){}

    Status InitParam() override {
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Output operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx, 
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        return Status::OK();
    }

    /**
    * \brief infer the shape of output and input.
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        return Status::OK();
    }


};

#ifdef USE_CUDA
INSTANCE_OUTPUT(NV, AK_FLOAT, Precision::FP32);
template class OutputHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_OUTPUT(X86, AK_FLOAT, Precision::FP32);
template class OutputHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_OUTPUT(ARM, AK_FLOAT, Precision::FP32);
template class OutputHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, ARM, AK_FLOAT, Precision::FP32);
#endif //arm

//! register op
ANAKIN_REGISTER_OP(Output)
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("output")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("output")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("output")
#endif
.Doc("Output operator [ only a input data holder and reshape ] ");
} /* namespace ops */

} /* namespace anakin */

#endif
