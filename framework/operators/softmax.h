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

#ifndef ANAKIN_OPERATOR_SOFTMAX_H
#define ANAKIN_OPERATOR_SOFTMAX_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/softmax.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class SoftmaxHelper;

/// pooling op
/**
 * \brief softmax implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Softmax : public Operator<Ttype, Dtype, Ptype> {
public:
    Softmax() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class SoftmaxHelper<Ttype, Dtype, Ptype>;
};

#define INSTANCE_SOFTMAX(Ttype, Dtype, Ptype) \
template<> \
void Softmax<Ttype, Dtype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<SoftmaxHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<SoftmaxHelper<Ttype, Dtype, Ptype>*>\
                  (this->_helper)->_param_softmax; \
    impl->_funcs_softmax(ins, outs, param, ctx); \
}

/**
 * \brief softmax helper class to implement softmax
 * public inherit OperatorHelper
 * including init resource and shape size in softmax context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class SoftmaxHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    SoftmaxHelper()=default;

    ~SoftmaxHelper() {}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing Softmax op parameter.";
        auto axis = GET_PARAMETER(int, axis);

        SoftmaxParam<Tensor4d<Ttype, Dtype>> param_softmax(axis);
        _param_softmax = param_softmax;
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for softmax operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, STATIC, SABER_IMPL, ctx));
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
        SABER_CHECK(_funcs_softmax.compute_output_shape(ins, outs, _param_softmax));
        return Status::OK();
    }

public:
    ///< _param_softmax stand for softmax parameter
    saber::SoftmaxParam<Tensor4d<Ttype, Dtype>> _param_softmax;
    ///< _funcs_softmax stand for softmax function 
    saber::Softmax<Ttype, Dtype> _funcs_softmax;
};

#ifdef USE_CUDA
INSTANCE_SOFTMAX(NV, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SOFTMAX(X86, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SOFTMAX(ARM, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Softmax)
.Doc("Softmax operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("softmax")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("softmax")
#endif
.num_in(1)
.num_out(1)
.Args<int>("axis", " axis ");

} /* namespace ops */

} /* namespace anakin */

#endif
