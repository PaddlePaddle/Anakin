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

#ifndef ANAKIN_OPERATOR_CRF_DECODING_H
#define ANAKIN_OPERATOR_CRF_DECODING_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/crf_decoding.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class CrfDecodingHelper;

/// pooling op
/**
 * \brief CrfDecoding operation class
 * public inheritance Operator
 */
template<typename Ttype, Precision Ptype>
class CrfDecoding : public Operator<Ttype, Ptype> {
public:
    CrfDecoding() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
    }

    friend class CrfDecodingHelper<Ttype, Ptype>;
};

/**
 * \brief CrfDecoding helper class
 * public inherit OperatorHelper
 * including init resource and shape size in crf_decoding context
 */
template<typename Ttype, Precision Ptype>
class CrfDecodingHelper : public OperatorHelper<Ttype, Ptype> {
public:
    CrfDecodingHelper()=default;

    ~CrfDecodingHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for CrfDecoding operation context
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
    ///< _param_crf_decoding stand for CrfDecoding parameter
    saber::CrfDecodingParam<Ttype>  _param_crf_decoding;
    ///< _funcs_crf_decoding stand for CrfDecoding function
    saber::CrfDecoding<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_crf_decoding;

private:
    ///< _dims stand for CrfDecoding size
    PTuple<int> _dims; 
};



} /* namespace ops */

} /* namespace anakin */

#endif
