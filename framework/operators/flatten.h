/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

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

#ifndef ANAKIN_OPERATOR_FLATTEN_H
#define ANAKIN_OPERATOR_FLATTEN_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/flatten.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class FlattenHelper;

//! pooling op
template<typename Ttype, DataType Dtype, Precision Ptype>
class Flatten : public Operator<Ttype, Dtype, Ptype> {
public:
    Flatten() {}

    //! forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator flatten<TargetType:" << "unknown" << ","
                   << type_id<typename DataTypeWarpper<Dtype>::type>().type_info() << ">";
    }

    friend class FlattenHelper<Ttype, Dtype, Ptype>;
};

#define INSTANCE_FLATTEN(Ttype, Dtype, Ptype) \
template<> \
void Flatten<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<FlattenHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<FlattenHelper<Ttype, Dtype, Ptype>*> \
                  (this->_helper)->_param_flatten; \
    impl->_funcs_flatten(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
class FlattenHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    FlattenHelper()=default;

    ~FlattenHelper() {}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing Flatten op parameter.";
        auto start_axis = GET_PARAMETER(int, start_axis);
        auto end_axis = GET_PARAMETER(int, end_axis);

        saber::FlattenParam<Tensor4d<Ttype, Dtype>> flatten_param;
        _param_flatten = flatten_param;
        return Status::OK();
    }

    //! initial all the resource needed by pooling
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_flatten.init(ins, outs, _param_flatten, SPECIFY, SABER_IMPL, ctx));
        return Status::OK();
    }

    //! infer the shape of output and input.
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_flatten.compute_output_shape(ins, outs, _param_flatten));
        return Status::OK();
    }

public:
    saber::FlattenParam<Tensor4d<Ttype, Dtype>> _param_flatten;
    saber::Flatten<Ttype, Dtype> _funcs_flatten;

private:
    PTuple<int> _dims; 
};

#ifdef USE_CUDA
INSTANCE_FLATTEN(NV, AK_FLOAT, Precision::FP32);
template class FlattenHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Flatten, FlattenHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_FLATTEN(X86, AK_FLOAT, Precision::FP32);
template class FlattenHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Flatten, FlattenHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_FLATTEN(ARM, AK_FLOAT, Precision::FP32);
template class FlattenHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Flatten, FlattenHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Flatten)
.Doc("Flatten operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("flatten")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("flatten")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("flatten")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */

#endif
