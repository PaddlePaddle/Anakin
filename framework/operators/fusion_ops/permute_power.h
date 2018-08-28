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

#ifndef ANAKIN_OPERATOR_PERMUTE_POWER_H
#define ANAKIN_OPERATOR_PERMUTE_POWER_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/permute_power.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class PermutePowerHelper;

/// pooling op
/**
 * \brief PermutePower implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class PermutePower : public Operator<Ttype, Ptype> {
public:
    PermutePower() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		LOG(ERROR) << "Not Impl Yet Operator PermutePower< Ttype("
				   << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	
    }

    friend class PermutePowerHelper<Ttype, Ptype>;
};

/**
 * \brief PermutePower helper class to implement it
 * public inherit OperatorHelper
 * including init resource and shape size in PermutePower context
 */
template<typename Ttype, Precision Ptype>
class PermutePowerHelper : public OperatorHelper<Ttype, Ptype> {
public:
    PermutePowerHelper()=default;

    ~PermutePowerHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for PermutePower operation context
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
    ///< _param_permute_power stand for PermutePower parameter
    saber::PermutePowerParam<Ttype> _param_permute_power;
    ///< _funcs_permute_power stand for PermutePower function
    saber::PermutePower<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_permute_power;

private:
    ///< _dims stand for PermutePower size
    PTuple<int> _dims; 
};



} /* namespace ops */

} /* namespace anakin */

#endif
