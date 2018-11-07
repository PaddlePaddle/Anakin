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

#ifndef ANAKIN_OPERATOR_RESHAPE_H
#define ANAKIN_OPERATOR_RESHAPE_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "framework/utils/layout_common.h"
#include "utils/logger/logger.h"
#include "saber/funcs/reshape.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class ReshapeHelper;

/// pooling op
/**
 * \brief Reshape implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class Reshape : public Operator<Ttype, Ptype> {
public:
    Reshape() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		LOG(ERROR) << "Not Impl Yet Operator Reshape< Ttype("
				   << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	
    }

    friend class ReshapeHelper<Ttype, Ptype>;
};


/**
 * \brief Reshape helper class to implement reshape
 * public inherit OperatorHelper
 * including init resource and shape size in reshape context
 */
template<typename Ttype, Precision Ptype>
class ReshapeHelper : public OperatorHelper<Ttype, Ptype> {
public:
    ReshapeHelper()=default;

    ~ReshapeHelper() {}

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by Reshape
    * \param ctx stand for reshape operation context
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
    ///< _param_reshape stand for reshape parameter
    saber::ReshapeParam<Ttype> _param_reshape;
    ///< _funcs_reshape stand for reshape function 
    saber::Reshape<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_reshape;
};

} /* namespace ops */

} /* namespace anakin */

#endif
