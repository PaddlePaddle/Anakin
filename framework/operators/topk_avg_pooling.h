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

#ifndef ANAKIN_OPERATOR_TOPK_AVG_POOLING_H
#define ANAKIN_OPERATOR_TOPK_AVG_POOLING_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/topk_avg_pooling.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class TopKAvgPoolingHelper;

/// pooling op
/**
 * \brief operation of ops class
 * public inheritance Operator
 */
template<typename Ttype, Precision Ptype>
class TopKAvgPooling : public Operator<Ttype, Ptype> {
public:
    TopKAvgPooling() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		LOG(ERROR) << "Not Impl Yet Operator TopKAvgPooling< Ttype("
				   << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	
    }

    friend class TopKAvgPoolingHelper<Ttype, Ptype>;
};

/**
 * \breif provide defined help for some operation
 *  public inheritance OperatorHelper
 *  including init operation context and the size of shape
 */
template<typename Ttype, Precision Ptype>
class TopKAvgPoolingHelper : public OperatorHelper<Ttype, Ptype> {
public:
    TopKAvgPoolingHelper()=default;

    ~TopKAvgPoolingHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for operation context
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
    ///< _param_topk_avg_pooling stand for topk_avg_pooling parameter
    saber::TopKAvgPoolingParam<Ttype> _param_topk_avg_pooling;
    ///< _funcs_topk_avg_pooling stand for topk_avg_pooling function
    saber::TopKAvgPooling<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_topk_avg_pooling;
};



} /* namespace ops */

} /* namespace anakin */

#endif
