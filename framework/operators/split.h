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

#ifndef ANAKIN_OPERATOR_SPLIT_H
#define ANAKIN_OPERATOR_SPLIT_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class SplitHelper;

/// pooling op
/**
 * \brief Split implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Split : public Operator<Ttype, Dtype, Ptype> {
public:
    Split() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        //LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   //<<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class SplitHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_SPLIT(Ttype, Dtype, Ptype) \
template<> \
void Split<Ttype, Dtype, Ptype>::operator()( \
        OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {}
/**
 * \brief Split helper class to implement Split
 * public inherit OperatorHelper
 * including init resource and shape size in Split context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class SplitHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    SplitHelper()=default;

    ~SplitHelper(){}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing Split op parameter.";
        split_num = GET_PARAMETER(int, split_num);
        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Split operation context
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
        for (int i = 0; i < split_num; i++) {
            outs[i]->set_shape(ins[0]->valid_shape());
            outs[i]->set_seq_offset(ins[0]->get_seq_offset());
        }
        return Status::OK();
    }

public:
    ///< split_num stand for split-numbers
    int split_num;

};

#ifdef USE_CUDA
INSTANCE_SPLIT(NV, AK_FLOAT, Precision::FP32);
template class SplitHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SPLIT(ARM, AK_FLOAT, Precision::FP32);
template class SplitHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SPLIT(X86, AK_FLOAT, Precision::FP32);
template class SplitHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, X86, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Split)
.Doc("Split operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("split")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("split")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("split")
#endif
.num_in(1)
.num_out(1)
.Args<int>("split_num", " split output number. ");

} /* namespace ops */

} /* namespace anakin */

#endif
