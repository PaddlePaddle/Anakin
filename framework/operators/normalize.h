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

#ifndef ANAKIN_OPERATOR_NORMALIZE_H
#define ANAKIN_OPERATOR_NORMALIZE_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/normalize.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class NormalizeHelper;

/// pooling op
/**
 * \brief Normalize operation class
 * public inheritance Operator
 */
template<typename Ttype, Precision Ptype>
class Normalize : public Operator<Ttype, Ptype> {
public:
    Normalize() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		LOG(ERROR) << "Not Impl Yet Operator Normalize< Ttype("
				   << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	
    }

    friend class NormalizeHelper<Ttype, Ptype>;
};

/**
 * \brief Normalize helper class 
 * public inherit OperatorHelper
 * including init resource and shape size in normalize context
 */
template<typename Ttype, Precision Ptype>
class NormalizeHelper : public OperatorHelper<Ttype, Ptype> {
public:
    NormalizeHelper()=default;

    ~NormalizeHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Normalize operation context
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
    ///< _param_normalize stand for Normalize parameter
    saber::NormalizeParam<Ttype>  _param_normalize;
    ///< _funcs_normalize stand for Normalize function
    saber::Normalize<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_normalize;

private:
    ///< _dims stand for Normalize size
    PTuple<int> _dims; 
};



} /* namespace ops */

} /* namespace anakin */

#endif
