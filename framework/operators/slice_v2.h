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

#ifndef ANAKIN_OPERATOR_SLICE_V2_H
#define ANAKIN_OPERATOR_SLICE_V2_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/slice_v2.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class SliceV2Helper;

/// pooling op
/**
 * \brief SliceV2 implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class SliceV2 : public Operator<Ttype, Ptype> {
public:
    SliceV2() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		LOG(ERROR) << "Not Impl Yet Operator SliceV2< Ttype("
				   << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	
    }

    friend class SliceV2Helper<Ttype, Ptype>;
};

/**
 * \brief SliceV2 helper class to implement SliceV2
 * public inherit OperatorHelper
 * including init resource and shape size in SliceV2 context
 */
template<typename Ttype, Precision Ptype>
class SliceV2Helper : public OperatorHelper<Ttype, Ptype> {
public:
    SliceV2Helper()=default;

    ~SliceV2Helper() {}

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for SliceV2 operation context
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
    ///< _param_slice_v2 stand for slice_v2 parameter
    saber::SliceV2Param<Ttype> _param_slice_v2;
    ///< _funcs_slice_v2 stand for slice_v2 function 
    saber::SliceV2<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_slice_v2;

};

} /* namespace ops */

} /* namespace anakin */

#endif
