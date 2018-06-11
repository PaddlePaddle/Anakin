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

#ifndef ANAKIN_OPERATOR_ELTWISE_H
#define ANAKIN_OPERATOR_ELTWISE_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/eltwise.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class EltwiseHelper;

/// pooling op
/**
 * \brief Eltwise implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Eltwise : public Operator<Ttype, Dtype, Ptype> {
public:
    Eltwise() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator convolution<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class EltwiseHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_ELTWISE(Ttype, Dtype, Ptype) \
template<> \
void Eltwise<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<EltwiseHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<EltwiseHelper<Ttype, Dtype, Ptype>*> \
                  (this->_helper)->_param_eltwise; \
    impl->_funcs_eltwise(ins, outs, param, ctx); \
}
/**
 * \brief Eltwise helper class to implement it
 * public inherit OperatorHelper
 * including init resource and shape size in Eltwise context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class EltwiseHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    EltwiseHelper()=default;

    ~EltwiseHelper() {}

    Status InitParam() override {
                DLOG(WARNING) << "Parsing Eltwise op parameter.";
        auto type = GET_PARAMETER(std::string, type);
        auto coeff = GET_PARAMETER(PTuple<float>, coeff);
        EltwiseType elt_type;

        if (type == "Add") {
            elt_type = Eltwise_sum;
        } else if (type == "Max") {
            elt_type = Eltwise_max;
        } else {
            elt_type = Eltwise_prod;
        }
        saber::EltwiseParam<Tensor4d<Ttype, Dtype> > eltwise_param(elt_type, coeff.vector());
        _param_eltwise = eltwise_param;
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Eltwise operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_eltwise.init(ins, outs, _param_eltwise, SPECIFY, SABER_IMPL, ctx));
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
        SABER_CHECK(_funcs_eltwise.compute_output_shape(ins, outs, _param_eltwise));
        return Status::OK();
    }

public:
    ///< _param_eltwise stand for Eltwise parameter
    saber::EltwiseParam<Tensor4d<Ttype, Dtype>>  _param_eltwise;
     ///< _funcs_eltwise stand for Eltwise function
    saber::Eltwise<Ttype, Dtype> _funcs_eltwise;

private:
    ///< _dims stand for Eltwise size
    PTuple<int> _dims; 
};

#ifdef USE_CUDA
INSTANCE_ELTWISE(NV, AK_FLOAT, Precision::FP32);
template class EltwiseHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Eltwise, EltwiseHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_ELTWISE(X86, AK_FLOAT, Precision::FP32);
template class EltwiseHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Eltwise, EltwiseHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ELTWISE(ARM, AK_FLOAT, Precision::FP32);
template class EltwiseHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Eltwise, EltwiseHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Eltwise)
.Doc("Eltwise operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("eltwise")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("eltwise")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("eltwise")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " eltwise type( string )")
.Args<PTuple<float>>("coeff", "coeff of eltwise");

} /* namespace ops */

} /* namespace anakin */

#endif
