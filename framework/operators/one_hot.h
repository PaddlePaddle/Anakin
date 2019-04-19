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

#ifndef ANAKIN_OPERATOR_ONE_HOT_H
#define ANAKIN_OPERATOR_ONE_HOT_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/one_hot.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class OneHotHelper;

/// pooling op
/**
 * \brief  operation class
 * public inheritance Operator
 */
template<typename Ttype, Precision Ptype>
class OneHot : public Operator<Ttype, Ptype> {
public:
    OneHot() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx,
            const std::vector<Tensor4dPtr<Ttype> >& ins,
            std::vector<Tensor4dPtr<Ttype> >& outs) {

        LOG(ERROR) << "Not Impl Yet Operator OneHot< Ttype("
        << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";
    }

    friend class OneHotHelper<Ttype, Ptype>;
};

/**
 * \brief  helper class
 * public inherit OperatorHelper
 * including init resource and shape size in one_hot context
 */
template<typename Ttype, Precision Ptype>
class OneHotHelper : public OperatorHelper<Ttype, Ptype> {
public:
    OneHotHelper() = default;

    ~OneHotHelper() = default;

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for one_hot operation context
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
    ///< _param_one_hot stand for one_hot parameter
    saber::OneHotParam<Ttype>  _param_one_hot;
    ///< _funcs_one_hot stand for one_hot function
    saber::OneHot<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_one_hot;

private:
    ///< _dims stand for OneHot size
    PTuple<int> _dims;
};

} /* namespace ops */

} /* namespace anakin */

#endif
