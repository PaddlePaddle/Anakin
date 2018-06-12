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

#ifndef ANAKIN_OPERATOR_RELU_H
#define ANAKIN_OPERATOR_RELU_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/activation.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class ReLUHelper;

/// pooling op
/**
 * \brief ReLU implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class ReLU : public Operator<Ttype, Dtype, Ptype> {
public:
    ReLU() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class ReLUHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_RELU(Ttype, Dtype, Ptype) \
template<> \
void ReLU<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,\
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<ReLUHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_relu; \
    impl->_funcs_relu(ins, outs, param, ctx); \
}
/**
 * \brief Relu helper class to implement Relu
 * public inherit OperatorHelper
 * including init resource and shape size in ReLU context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class ReLUHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    ReLUHelper()=default;

    ~ReLUHelper() {}

    Status InitParam() override {
                DLOG(WARNING) << "Parsing ReLU op parameter.";

        // get relu param
        auto alpha = GET_PARAMETER(float, alpha);
        ActivationParam<Tensor4d<Ttype, Dtype>> active_param(Active_relu);//, alpha); // TEMP
        _param_relu = active_param;
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for ReLU operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_relu.init(ins, outs, _param_relu, SPECIFY, VENDER_IMPL, ctx));
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
        SABER_CHECK(_funcs_relu.compute_output_shape(ins, outs, _param_relu));
        return Status::OK();
    }

public:
    ///< _param_relu stand for ReLU parameter
    saber::ActivationParam<Tensor4d<Ttype, Dtype>> _param_relu;
    ///< _funcs_relu stand for ReLU function 
    saber::Activation<Ttype, Dtype> _funcs_relu;

private:
    ///< _dims stand for ReLU size
    PTuple<int> _dims; 
};

#ifdef USE_CUDA
INSTANCE_RELU(NV, AK_FLOAT, Precision::FP32);
template class ReLUHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReLU, ReLUHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_RELU(X86, AK_FLOAT, Precision::FP32);
template class ReLUHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReLU, ReLUHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_RELU(ARM, AK_FLOAT, Precision::FP32);
template <>
Status ReLUHelper<ARM, AK_FLOAT, Precision::FP32>::Init(OpContext<ARM> &ctx,
            const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins,
            std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_relu.init(ins, outs, _param_relu, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(ReLU, ReLUHelper, ARM, AK_FLOAT, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(ReLU)
.Doc("ReLU operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("Relu")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("Relu")
#endif
#ifdef USE_X86_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("Relu")
#endif
.num_in(1)
.num_out(1)
.Args<float>("alpha", " alpha for relu");

} /* namespace ops */

} /* namespace anakin */

#endif
