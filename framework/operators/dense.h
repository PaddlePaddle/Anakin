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

#ifndef ANAKIN_OPERATOR_DENSE_H
#define ANAKIN_OPERATOR_DENSE_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/fc.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class DenseHelper;

/// pooling op
/**
 * \brief Dense operation class
 * public inheritance Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Dense : public Operator<Ttype, Dtype, Ptype> {
public:
    Dense() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator convolution<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class DenseHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_DENSE(Ttype, Dtype, Ptype) \
template<> \
void Dense<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<DenseHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<DenseHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_dense; \
    impl->_funcs_dense(ins, outs, param, ctx); \
}
/**
 * \brief Dense helper class 
 * public inherit OperatorHelper
 * including init resource and shape size in dense context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class DenseHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    DenseHelper()=default;

    ~DenseHelper() {}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing Dense op parameter.";
        auto axis = GET_PARAMETER(int, axis);
        auto out_dim = GET_PARAMETER(int, out_dim);
        auto bias_term = GET_PARAMETER(bool, bias_term);

        auto weights = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_1);

        if (bias_term) {
            auto bias = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_2);
            saber::FcParam<Tensor4d<Ttype, Dtype>> fc_param(&(weights.d_tensor()), &(bias.d_tensor()), out_dim,
                                                            axis);
            _param_dense = fc_param;
        } else {
            Tensor4d<Ttype, Dtype>* bias = nullptr;
            saber::FcParam<Tensor4d<Ttype, Dtype>> fc_param(&(weights.d_tensor()), bias, out_dim, axis);
            _param_dense = fc_param;
        }
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Dense operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_dense.init(ins, outs, _param_dense, STATIC, VENDER_IMPL, ctx));
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
        SABER_CHECK(_funcs_dense.compute_output_shape(ins, outs, _param_dense));
        return Status::OK();
    }

public:
    ///< _param_dense stand for Dense parameter
    saber::FcParam<Tensor4d<Ttype, Dtype>>  _param_dense;
    ///< _funcs_dense stand for Dense function
    saber::Fc<Ttype, Dtype> _funcs_dense;

private:
    ///< _dims stand for Dense size
    PTuple<int> _dims; 
};

#ifdef USE_CUDA
INSTANCE_DENSE(NV, AK_FLOAT, Precision::FP32);
template class DenseHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Dense, DenseHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_DENSE(ARM, AK_FLOAT, Precision::FP32);
template<>
Status DenseHelper<ARM, AK_FLOAT, Precision::FP32>::Init(OpContext<ARM> &ctx,\
        const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins, \
                std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_dense.init(ins, outs, _param_dense, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Dense, DenseHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_DENSE(X86, AK_FLOAT, Precision::FP32);
template class DenseHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Dense, DenseHelper, X86, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Dense)
.Doc("Dense operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("fullconnect")
.__alias__<NV, AK_FLOAT, Precision::FP32>("fc")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("fullconnect")
.__alias__<ARM, AK_FLOAT, Precision::FP32>("fc")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("fullconnect")
.__alias__<X86, AK_FLOAT, Precision::FP32>("fc")
#endif
.num_in(1)
.num_out(1)
.Args<int>("axis", " axis to compute ")
.Args<int>("out_dim", " out dim ")
.Args<bool>("bias_term", " whether fc weights have bias");

} /* namespace ops */

} /* namespace anakin */

#endif
