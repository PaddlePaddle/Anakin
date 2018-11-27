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

#ifndef ANAKIN_OPERATOR_RELU_H
#define ANAKIN_OPERATOR_RELU_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/activation.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class ReLUHelper;

/// pooling op
/**
 * \brief ReLU implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class ReLU : public Operator<Ttype, Ptype> {
public:
    ReLU() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		LOG(ERROR) << "Not Impl Yet Operator ReLU< Ttype("
				   << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	
    }

    friend class ReLUHelper<Ttype, Ptype>;
};

/**
 * \brief Relu helper class to implement Relu
 * public inherit OperatorHelper
 * including init resource and shape size in ReLU context
 */
template<typename Ttype, Precision Ptype>
class ReLUHelper : public OperatorHelper<Ttype, Ptype> {
public:
    ReLUHelper()=default;

    ~ReLUHelper() {}

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for ReLU operation context
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
    ///< _param_relu stand for ReLU parameter
    saber::ActivationParam<Ttype> _param_relu;
    ///< _funcs_relu stand for ReLU function 
    saber::Activation<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_relu;

private:
    ///< _dims stand for ReLU size
    PTuple<int> _dims; 
};

} /* namespace ops */

} /* namespace anakin */

#endif
