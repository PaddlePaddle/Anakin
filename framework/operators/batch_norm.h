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

#ifndef ANAKIN_OPERATOR_BATCH_NORM_H
#define ANAKIN_OPERATOR_BATCH_NORM_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
//#include "saber/funcs/permute.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class BatchNormHelper;

/// pooling op
/**
 * \brief Batch normalization class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class BatchNorm : public Operator<Ttype, Dtype, Ptype> {
public:
    BatchNorm() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class BatchNormHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_BATCHNORM(Ttype, Dtype, Ptype) \
template<> \
void BatchNorm<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { }
/**
 * \brief Batch normalization helper class
 * public inherit OperatorHelper
 * including init resource and shape size in BatchNorm processing
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class BatchNormHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    BatchNormHelper()=default;

    ~BatchNormHelper() {}

    Status InitParam() override {
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for batchNorm operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
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
        for (int i = 0; i <  outs.size(); i++) {
            // set tensor shape tensor->set_shape(shape[i]);
            outs[i]->set_shape(ins[i]->shape());
        }

        return Status::OK();
    }

public:
    //PermuteParam<void> _param_permute;
    //saber::Permute<Ttype, Dtype> _funcs_permute;

private:
    ///< _dims stand for batchNorm size 
    PTuple<int> _dims; 
};


#ifdef USE_CUDA
INSTANCE_BATCHNORM(NV, AK_FLOAT, Precision::FP32);
template class BatchNormHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_BATCHNORM(X86, AK_FLOAT, Precision::FP32);
template class BatchNormHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_BATCHNORM(ARM, AK_FLOAT, Precision::FP32);
template class BatchNormHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(BatchNorm)
.Doc("BatchNorm operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("batchnorm")
#endif

#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("batchnorm")
#endif

#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("batchnorm")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims for permuting the order of input ");

} /* namespace ops */

} /* namespace anakin */

#endif
