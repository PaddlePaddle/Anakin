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

#ifndef ANAKIN_OPERATOR_IM2SEQUENCE_H
#define ANAKIN_OPERATOR_IM2SEQUENCE_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator_attr.h"
#include "utils/logger/logger.h"
#include "saber/funcs/im2sequence.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class Im2SequenceHelper;


/// im2sequence op
/**
 * \brief Im2Sequence implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Im2Sequence : public Operator<Ttype, Dtype, Ptype> {
public:
    Im2Sequence() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator Im2Sequence<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class Im2SequenceHelper<Ttype, Dtype, Ptype>;
};

/**
 * \brief Im2Sequence helper class to implement Im2Sequence 
 * public inherit OperatorHelper
 * including init resource and shape size in Im2Sequence context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Im2SequenceHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    Im2SequenceHelper()=default;

    ~Im2SequenceHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by im2sequence
    * \param ctx stand for Im2Sequence operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

    /**
    * \brief infer the shape of output and input.
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

public:
    ///< _param_im2sequence stand for Im2Sequence parameter
    saber::Im2SequenceParam<Tensor4d<Ttype, Dtype>> _param_im2sequence;
    ///< _funcs_im2sequence stand for Im2Sequence function
    saber::Im2Sequence<Ttype, Dtype> _funcs_im2sequence;
};

} /* namespace ops */

} /* namespace anakin */

#endif
