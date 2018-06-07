/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_FRAMEWORK_OPERATOR_ARGMAX_H
#define ANAKIN_FRAMEWORK_OPERATOR_ARGMAX_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator_attr.h"
#include "utils/logger/logger.h"
#include "saber/funcs/argmax.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class ArgmaxHelper;

/// axpy op
/**
 * \brief operation of argMax class
 * public inheritance Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Argmax : public Operator<Ttype, Dtype, Ptype> {
public:
    Argmax() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator Argmax<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class ArgmaxHelper<Ttype, Dtype, Ptype>;
};

/**
 * \breif provide defined help for some operation
 *  public inheritance OperatorHelper
 *  including init operation context and the size of shape
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class ArgmaxHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    ArgmaxHelper()=default;

    ~ArgmaxHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by argmax
    * \param ctx stand for operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

    /**
    * \brief infer the shape of output and input.
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

public:
    ///< _param_argmax stand for argmax parameter
    saber::ArgmaxParam<Tensor4d<Ttype, Dtype>> _param_argmax;
    ///< _funcs_argmax stand for argmax function
    saber::Argmax<Ttype, Dtype> _funcs_argmax;

private:
    ///< _dims stand for argmax size
    PTuple<int> _dims; 
};



} /* namespace ops */

} /* namespace anakin */

#endif
