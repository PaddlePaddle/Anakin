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

#ifndef ANAKIN_OPERATOR_RESHAPE_H
#define ANAKIN_OPERATOR_RESHAPE_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/reshape.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class ReshapeHelper;

/// pooling op
/**
 * \brief Reshape implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Reshape : public Operator<Ttype, Dtype, Ptype> {
public:
    Reshape() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class ReshapeHelper<Ttype, Dtype, Ptype>;
};

#define INSTANCE_RESHAPE(Ttype, Dtype, Ptype) \
template<> \
void Reshape<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<ReshapeHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ReshapeHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_reshape; \
    impl->_funcs_reshape(ins, outs, param, ctx); \
}
/**
 * \brief Reshape helper class to implement reshape
 * public inherit OperatorHelper
 * including init resource and shape size in reshape context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class ReshapeHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    ReshapeHelper()=default;

    ~ReshapeHelper() {}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing Reshape op parameter.";
        auto dims = GET_PARAMETER(PTuple<int>, dims);

        ReshapeParam<Tensor4d<Ttype, Dtype>> param_reshape(dims.vector());
        _param_reshape = param_reshape;
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by Reshape
    * \param ctx stand for reshape operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_reshape.init(ins, outs, _param_reshape, SPECIFY, SABER_IMPL, ctx));
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
        SABER_CHECK(_funcs_reshape.compute_output_shape(ins, outs, _param_reshape));
        outs[0]->set_seq_offset(ins[0]->get_seq_offset());
        return Status::OK();
    }

public:
    ///< _param_reshape stand for reshape parameter
    saber::ReshapeParam<Tensor4d<Ttype, Dtype>> _param_reshape;
    ///< _funcs_reshape stand for reshape function 
    saber::Reshape<Ttype, Dtype> _funcs_reshape;
};

#ifdef USE_CUDA
INSTANCE_RESHAPE(NV, AK_FLOAT, Precision::FP32);
template class ReshapeHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_RESHAPE(X86, AK_FLOAT, Precision::FP32);
template class ReshapeHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_RESHAPE(ARM, AK_FLOAT, Precision::FP32);
template class ReshapeHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Reshape)
.Doc("Reshape operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("reshape")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("reshape")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("reshape")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims of redhape target");

} /* namespace ops */

} /* namespace anakin */

#endif
