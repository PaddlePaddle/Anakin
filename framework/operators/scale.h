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

#ifndef ANAKIN_OPERATOR_SCALE_H
#define ANAKIN_OPERATOR_SCALE_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/scale.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class ScaleHelper;

/// pooling op
/**
 * \brief operation of ops class
 * public inheritance Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Scale : public Operator<Ttype, Dtype, Ptype> {
public:
    Scale() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class ScaleHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_SCALE(Ttype, Dtype, Ptype) \
template<> \
void Scale<Ttype, Dtype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<ScaleHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ScaleHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_scale; \
    impl->_funcs_scale(ins, outs, param, ctx); \
}
/**
 * \breif provide defined help for some operation
 *  public inheritance OperatorHelper
 *  including init operation context and the size of shape
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class ScaleHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    ScaleHelper()=default;

    ~ScaleHelper(){}

    Status InitParam() override {
                DLOG(WARNING) << "Parsing Scale op parameter.";
        auto axis = GET_PARAMETER(int, axis);
        auto num_axes = GET_PARAMETER(int, num_axes);
        auto bias_term = GET_PARAMETER(bool, bias_term);
        auto weights = GET_PARAMETER(PTuple<typename DataTypeWarpper<Dtype>::type>, weight_1);
        auto bias = GET_PARAMETER(PTuple<typename DataTypeWarpper<Dtype>::type>, weight_2);
        ScaleParam<Tensor4d<Ttype, Dtype>> param_scale(weights.vector(), bias.vector(), bias_term, axis, num_axes);
        _param_scale = param_scale;
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_scale.init(ins, outs, _param_scale, SPECIFY, VENDER_IMPL, ctx));
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
        SABER_CHECK(_funcs_scale.compute_output_shape(ins, outs, _param_scale));
        return Status::OK();
    }

public:
    ///< _param_scale stand for scale parameter
    saber::ScaleParam<Tensor4d<Ttype, Dtype>> _param_scale;
    ///< _funcs_scale stand for scale function
    saber::Scale<Ttype, Dtype> _funcs_scale;
};

#ifdef USE_CUDA
INSTANCE_SCALE(NV, AK_FLOAT, Precision::FP32);
template class ScaleHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SCALE(X86, AK_FLOAT, Precision::FP32);
template class ScaleHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SCALE(ARM, AK_FLOAT, Precision::FP32);
template <>
Status ScaleHelper<ARM, AK_FLOAT, Precision::FP32>::Init(OpContext<ARM> &ctx, \
    const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins, \
    std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_scale.init(ins, outs, _param_scale, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}

ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, ARM, AK_FLOAT, Precision::FP32);
#endif//arm


//! register op
ANAKIN_REGISTER_OP(Scale)
.Doc("Scale operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("Scale")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("Scale")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("Scale")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of Scale ");


} /* namespace ops */

} /* namespace anakin */

#endif
