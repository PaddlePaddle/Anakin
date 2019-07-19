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

#ifndef ANAKIN_OPERATOR_SEQUENCE_POOL_H
#define ANAKIN_OPERATOR_SEQUENCE_POOL_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/sequence_pool.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class SequencePoolHelper;

/// pooling op
/**
 * \brief SequencePool operation class
 * public inheritance Operator
 */
template<typename Ttype, Precision Ptype>
class SequencePool : public Operator<Ttype, Ptype> {
public:
    SequencePool() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		//LOG(ERROR) << "Not Impl Yet Operator SequencePool< Ttype(" 
                     //<< target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	
    }

    friend class SequencePoolHelper<Ttype, Ptype>;
};

/**
 * \brief SequencePool helper class
 * public inherit OperatorHelper
 * including init resource and shape size in sequence_pool context
 */
template<typename Ttype, Precision Ptype>
class SequencePoolHelper : public OperatorHelper<Ttype, Ptype> {
public:
    SequencePoolHelper()=default;

    ~SequencePoolHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for SequencePool operation context
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
    ///< _param_sequence_pool stand for SequencePool parameter
    saber::SequencePoolParam<Ttype>  _param_sequence_pool;
    ///< _funcs_sequence_pool stand for SequencePool function
    saber::SequencePool<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_sequence_pool;

private:
    ///< _dims stand for SequencePool size
    PTuple<int> _dims; 
};

} /* namespace ops */

} /* namespace anakin */

#endif
