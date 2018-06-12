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

#ifndef ANAKIN_OPERATOR_NORMALIZE_H
#define ANAKIN_OPERATOR_NORMALIZE_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/layer_norm.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class LayerNormHelper;

/// pooling op
/**
 * \brief Normalize operation class
 * public inheritance Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class LayerNorm : public Operator<Ttype, Dtype, Ptype> {
public:
    LayerNorm() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator Normalize<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class LayerNormHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_LAYERNORM(Ttype, Dtype, Ptype) \
template<> \
void LayerNorm<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<LayerNormHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<LayerNormHelper<Ttype, Dtype, Ptype>*> \
                  (this->_helper)->_param_layer_norm; \
    impl->_funcs_layer_norm(ins, outs, param, ctx); \
}
/**
 * \brief Normalize helper class 
 * public inherit OperatorHelper
 * including init resource and shape size in normalize context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class LayerNormHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    LayerNormHelper()=default;

    ~LayerNormHelper() {}

    Status InitParam() override {
        auto axis = GET_PARAMETER(int, begin_norm_axis);
        auto eps = GET_PARAMETER(float, eps);

        auto input_scale = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_1);
        auto input_bias = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_2);

        saber::LayerNormParam<Tensor4d<Ttype, Dtype>> param(axis, eps, &(input_scale.d_tensor()), \
            &(input_bias.d_tensor()));
        _param_layer_norm = param;
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Normalize operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_layer_norm.init(ins, outs, _param_layer_norm, SPECIFY, SABER_IMPL, ctx));
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
        SABER_CHECK(_funcs_layer_norm.compute_output_shape(ins, outs, _param_layer_norm));
        return Status::OK();
    }

public:
    ///< _param_normalize stand for Normalize parameter
    saber::LayerNormParam<Tensor4d<Ttype, Dtype>>  _param_layer_norm;
    ///< _funcs_normalize stand for Normalize function
    saber::LayerNorm<Ttype, Dtype> _funcs_layer_norm;
};

#ifdef USE_CUDA
INSTANCE_LAYERNORM(NV, AK_FLOAT, Precision::FP32);
template class LayerNormHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_LAYERNORM(X86, AK_FLOAT, Precision::FP32);
template class LayerNormHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_LAYERNORM(ARM, AK_FLOAT, Precision::FP32);
template class LayerNormHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(LayerNorm)
.Doc("LayerNorm operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("layernorm")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("layernorm")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("layernorm")
#endif
.num_in(1)
.num_out(1)
.Args<int>("begin_norm_axis", " begin norm axis")
.Args<float>("eps", "eps");

} /* namespace ops */

} /* namespace anakin */

#endif
