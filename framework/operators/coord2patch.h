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

#ifndef ANAKIN_OPERATOR_COORD2PATCH_H
#define ANAKIN_OPERATOR_COORD2PATCH_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/coord2patch.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class Coord2PatchHelper;

/// pooling op
/**
 * \brief Coord2Patch implementation class
 * public inherit Operator
 */
template<typename Ttype, Precision Ptype>
class Coord2Patch : public Operator<Ttype, Ptype> {
public:
    Coord2Patch() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx,
                             const std::vector<Tensor4dPtr<Ttype> >& ins,
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator Convolution< Ttype("
           << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";
    }

    friend class Coord2PatchHelper<Ttype, Ptype>;
};

/**
 * \brief Permut helper class to implement conv 3X3
 * public inherit OperatorHelper
 * including init resource and shape size in Permut context
 */
template<typename Ttype, Precision Ptype>
class Coord2PatchHelper : public OperatorHelper<Ttype, Ptype> {
public:
    Coord2PatchHelper()=default;

    ~Coord2PatchHelper() {}

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Permut operation context
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
    ///< _param_coord2patch stand for Coord2Patch parameter
    saber::Coord2PatchParam<Ttype> _param_coord2patch;
    ///< _funcs_coord2patch stand for Coord2Patch function
    saber::Coord2Patch<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_coord2patch;
};

} /* namespace ops */

} /* namespace anakin */

#endif//ANAKIN_OPERATOR_COORD2PATCH_H
