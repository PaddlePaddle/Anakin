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

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Pooling operation context
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
    ///< _param_pooling stand for Pooling parameter
    saber::PoolingParam<Tensor4d<Ttype, Dtype>> _param_pooling;
    ///< _funcs_pooling stand for Pooling function
    saber::Pooling<Ttype, Dtype> _funcs_pooling;
};


} /* namespace ops */

} /* namespace anakin */

#endif
