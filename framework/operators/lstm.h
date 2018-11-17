

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

#ifndef ANAKIN_OPERATOR_LSTM_H
#define ANAKIN_OPERATOR_LSTM_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/lstm.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class LstmHelper;


/// lstm op
/**
 * \brief Lstm implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class Lstm : public Operator<Ttype, Ptype> {
public:
    Lstm() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		LOG(ERROR) << "Not Impl Yet Operator Lstm< Ttype("
				   << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	

    }

    friend class LstmHelper<Ttype, Ptype>;
};

/**
 * \brief Lstm helper class to implement Lstm
 * public inherit OperatorHelper
 * including init resource and shape size in Lstm context
 */
template<typename Ttype, Precision Ptype>
class LstmHelper : public OperatorHelper<Ttype, Ptype> {
public:
    LstmHelper()=default;

    ~LstmHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by lstm
    * \param ctx stand for Lstm operation context
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
    ///< _param_lstm stand for Lstm parameter
    saber::LstmParam<Ttype> _param_lstm;
    ///< _funcs_lstm stand for Lstm function
    saber::Lstm<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_lstm;
};

} /* namespace ops */

} /* namespace anakin */

#endif


