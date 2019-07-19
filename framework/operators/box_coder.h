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

#ifndef ANAKIN_FRAMEWORK_OPERATORS_BOX_CODER_H
#define ANAKIN_FRAMEWORK_OPERATORS_BOX_CODER_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/box_coder.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class BoxCoderHelper;

/// axpy op
/**
 * \brief operation of BoxCoder class
 * public inheritance Operator
 */
template<typename Ttype, Precision Ptype>
class BoxCoder : public Operator<Ttype, Ptype> {
public:
    BoxCoder() {}

    /// forward impl
    virtual void operator()(OpContext<Ttype>& ctx,
                            const std::vector<Tensor4dPtr<Ttype> >& ins,
                            std::vector<Tensor4dPtr<Ttype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator BoxCoder< Ttype("
                   << target_name<Ttype>::value << "), Precision(" << Ptype << ") >";
    }

    friend class BoxCoderHelper<Ttype, Ptype>;
};

/**
 * \breif provide defined help for some operation
 *  public inheritance OperatorHelper
 *  including init operation context and the size of shape
 */
template<typename Ttype, Precision Ptype>
class BoxCoderHelper : public OperatorHelper<Ttype, Ptype> {
public:
    BoxCoderHelper() = default;

    ~BoxCoderHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by BoxCoder
    * \param ctx stand for operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype>& ctx,
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
    ///< _param_axpy stand for axpy parameter
    saber::BoxCoderParam<Ttype> _param_box_coder;
    ///< _funcs_box_coder stand for axpy function
    saber::BoxCoder<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_box_coder;

private:
    ///< _dims stand for axpy size
    PTuple<int> _dims;
    Tensor<Ttype> _tensor_var;
};



} /* namespace ops */

} /* namespace anakin */

#endif
