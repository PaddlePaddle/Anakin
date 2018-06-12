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

#ifndef ANAKIN_OPERATOR_CONCAT_H
#define ANAKIN_OPERATOR_CONCAT_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/concat.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class ConcatHelper;

/// pooling op
/**
 * \brief contct class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Concat : public Operator<Ttype, Dtype, Ptype> {
public:
    Concat() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class ConcatHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_CONCAT(Ttype, Dtype, Ptype) \
template<> \
void Concat<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<ConcatHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ConcatHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_concat; \
    impl->_funcs_concat(ins, outs, param, ctx); \
}
/**
 * \brief contact helper class
 * public inherit OperatorHelper 
 * including init resource and shape size in contact context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class ConcatHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    ConcatHelper()=default;

    ~ConcatHelper() {}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing Concat op parameter.";
        auto axis = GET_PARAMETER(int, axis);
        ConcatParam<Tensor4d<Ttype, Dtype>> param_concat(axis);
        _param_concat = param_concat;
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for contact operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_concat.init(ins, outs, _param_concat, SPECIFY, SABER_IMPL, ctx));
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
        SABER_CHECK(_funcs_concat.compute_output_shape(ins, outs, _param_concat));
        return Status::OK();
    }

public:
    ///< _param_concat stand for contact parameter
    saber::ConcatParam<Tensor4d<Ttype, Dtype>> _param_concat;
    ///< _funcs_concat stand for contact function
    saber::Concat<Ttype, Dtype> _funcs_concat;

private:
    ///< _dims stand for contact size
    PTuple<int> _dims; 
};

#ifdef USE_CUDA
INSTANCE_CONCAT(NV, AK_FLOAT, Precision::FP32);
template class ConcatHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONCAT(ARM, AK_FLOAT, Precision::FP32);
template class ConcatHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONCAT(X86, AK_FLOAT, Precision::FP32);
template class ConcatHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, X86, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Concat)
.Doc("Concat operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("concat")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("concat")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("concat")
#endif
.num_in(2)
.num_out(1)
.Args<int>("axis", " axis for concat the input ");


} /* namespace ops */

} /* namespace anakin */

#endif
