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

#ifndef ANAKIN_OPERATOR_DECONV_H
#define ANAKIN_OPERATOR_DECONV_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/deconv.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class DeconvolutionHelper;

/// pooling op
/**
 * \brief Deconvolution operation class
 * public inheritance Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Deconvolution : public Operator<Ttype, Dtype, Ptype> {
public:
    Deconvolution() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator convolution<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class DeconvolutionHelper<Ttype, Dtype, Ptype>;
};

/**
 * \brief Deconvlution helper class 
 * public inherit OperatorHelper
 * including init resource and shape size in deconvolution context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class DeconvolutionHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    DeconvolutionHelper()=default;

    ~DeconvolutionHelper(){}

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for deconvolution operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Deconvolution operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

public:
    ///< _param_deconv stand for deconvolution parameter
    saber::ConvParam<Tensor4d<Ttype, Dtype>>  _param_deconv;
    ///< _funcs_deconv stand for deconvolution function
    saber::Deconv<Ttype, Dtype> _funcs_deconv;

private:
    ///< _dims stand for batchNorm size
    PTuple<int> _dims; 
};

} /* namespace ops */

} /* namespace anakin */

#endif
