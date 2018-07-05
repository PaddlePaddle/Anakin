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

#ifndef ANAKIN_OPERATOR_LAYER_NORM_H
#define ANAKIN_OPERATOR_LAYER_NORM_H

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

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Normalize operation context
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
    ///< _param_normalize stand for Normalize parameter
    saber::LayerNormParam<Tensor4d<Ttype, Dtype>>  _param_layer_norm;
    ///< _funcs_normalize stand for Normalize function
    saber::LayerNorm<Ttype, Dtype> _funcs_layer_norm;
};

} /* namespace ops */

} /* namespace anakin */

#endif //ANAKIN_OPERATOR_LAYER_NORM_H
