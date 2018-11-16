/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_OPERATOR_CONV_BATCHNORM_H
#define ANAKIN_OPERATOR_CONV_BATCHNORM_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/conv.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class ConvBatchnormHelper;

/// pooling op
/**
 * \brief ConvBatchnormHelper implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class ConvBatchnorm : public Operator<Ttype, Ptype> {
public:
    ConvBatchnorm() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		LOG(ERROR) << "Not Impl Yet Operator ConvBatchnorm< Ttype("
				   << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	
    }

    friend class ConvBatchnormHelper<Ttype, Ptype>;
};

/**
 * \brief ConvBatchnorm helper class to implement it
 * public inherit OperatorHelper
 * including init resource and shape size in ConvBatchnormHelper context
 */
template<typename Ttype, Precision Ptype>
class ConvBatchnormHelper : public OperatorHelper<Ttype, Ptype> {
public:
    ConvBatchnormHelper()=default;

    ~ConvBatchnormHelper() {}

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for ConvBatchnorm operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype> >& ins, 
                std::vector<Tensor4dPtr<Ttype> >& outs) override;

    /**
    * \brief infer the shape of output and input.
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                      std::vector<Tensor4dPtr<Ttype> >& outs) override;

public:
    ///< _param_conv_batchnorm stand for ConvBatchnorm parameter
    saber::ConvParam<Ttype>  _param_conv_batchnorm;
    ///< _funcs_conv stand for ConvBatchnorm function 
    saber::Conv<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_conv_batchnorm;
};

} /* namespace ops */

} /* namespace anakin */

#endif
