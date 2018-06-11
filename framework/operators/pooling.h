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

#ifndef ANAKIN_OPERATOR_POOLING_H
#define ANAKIN_OPERATOR_POOLING_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/pooling.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class PoolingHelper;

/**
* \brief PoolingType used for pooling operation
* pooling operation includes MAX, AVG,SUM
*/
enum class PoolingType {
    MAX,    ///< MAX stand for max-pooling operation
    AVG,    ///< AVG stand for avg-pooling operation
    SUM,    ///< SUM stand for sum-pooling operation 
};

/// pooling op
/**
 * \brief Pooling implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Pooling : public Operator<Ttype, Dtype, Ptype> {
public:
    Pooling() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator Pooling<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class PoolingHelper<Ttype, Dtype, Ptype>;
};

/// TODO ... specialization other type of operator
#define INSTANCE_POOLING(Ttype, Dtype, Ptype) \
template<> \
void Pooling<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<PoolingHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PoolingHelper<Ttype, Dtype, Ptype>*>\
                  (this->_helper)->_param_pooling; \
    impl->_funcs_pooling(ins, outs, param, ctx); \
}

/**
 * \brief Pooling helper class to implement Pooling 
 * public inherit OperatorHelper
 * including init resource and shape size in Pooling context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class PoolingHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    PoolingHelper()=default;

    ~PoolingHelper() {}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing Pooling op parameter.";
        auto cmp_out_shape_floor_as_conv = GET_PARAMETER(bool, cmp_out_shape_floor_as_conv);
        auto global_pooling = GET_PARAMETER(bool, global_pooling);
        auto pool_padding = GET_PARAMETER(PTuple<int>, padding);
        auto pool_strides = GET_PARAMETER(PTuple<int>, strides);
        auto pool_size = GET_PARAMETER(PTuple<int>, pool_size);
        auto pool_method = GET_PARAMETER(std::string, method);

        if (pool_method == "MAX") {
            PoolingParam<Tensor4d<Ttype, Dtype>> pooling_param(pool_size[0], pool_size[1],
                                                               pool_padding[0], pool_padding[1],
                                                               pool_strides[0], pool_strides[1],
                                                               Pooling_max, global_pooling, cmp_out_shape_floor_as_conv);
            _param_pooling = pooling_param;
        } else if (pool_method == "AVG") {
            PoolingParam<Tensor4d<Ttype, Dtype>> pooling_param(pool_size[0], pool_size[1],
                                                               pool_padding[0], pool_padding[1],
                                                               pool_strides[0], pool_strides[1],
                                                               Pooling_average_include_padding, global_pooling, cmp_out_shape_floor_as_conv);
            _param_pooling = pooling_param;
        } else {
                    LOG(FATAL) << " Pooling op doesn't support : " << pool_method << " pooling.";
        }

        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Pooling operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_pooling.init(ins, outs, _param_pooling, SPECIFY, VENDER_IMPL, ctx));
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
        SABER_CHECK(_funcs_pooling.compute_output_shape(ins, outs, _param_pooling));
        return Status::OK();
    }

public:
    ///< _param_pooling stand for Pooling parameter
    saber::PoolingParam<Tensor4d<Ttype, Dtype>> _param_pooling;
    ///< _funcs_pooling stand for Pooling function
    saber::Pooling<Ttype, Dtype> _funcs_pooling;
};

#ifdef USE_CUDA
INSTANCE_POOLING(NV, AK_FLOAT, Precision::FP32);
template class PoolingHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pooling, PoolingHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE

INSTANCE_POOLING(ARM, AK_FLOAT, Precision::FP32);
template <>
Status PoolingHelper<ARM, AK_FLOAT, Precision::FP32>::Init(OpContext<ARM> &ctx, \
    const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins, \
    std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_pooling.init(ins, outs, _param_pooling, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Pooling, PoolingHelper, ARM, AK_FLOAT, Precision::FP32);

#endif  //arm

#ifdef USE_X86_PLACE
INSTANCE_POOLING(X86, AK_FLOAT, Precision::FP32);
template class PoolingHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Pooling, PoolingHelper, X86, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Pooling)
.Doc("Pooling operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("pooling")
.__alias__<NV, AK_FLOAT, Precision::FP32>("pool")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("pooling")
.__alias__<ARM, AK_FLOAT, Precision::FP32>("pool")
#endif
#ifdef USE_X86_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("pooling")
.__alias__<ARM, AK_FLOAT, Precision::FP32>("pool")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("method", "Pooling type to be applied (MAX, SUM, AVG).")
.Args<bool>("cmp_out_shape_floor_as_conv cmp_out_shape_floor_as_conv of pooling for adu novel approach")
.Args<bool>("global_pooling", "whether execute global pooling on input")
.Args<PTuple<int>>("pool_size", " kernel size for pooling (x, y) or (x, y, z).")
.Args<PTuple<int>>("strides",  "stride for pooling (x, y)  or  (x, y, z).")
.Args<PTuple<int>>("padding", "pad for pooling: (x, y) or (x, y, z).");

} /* namespace ops */

} /* namespace anakin */

#endif
