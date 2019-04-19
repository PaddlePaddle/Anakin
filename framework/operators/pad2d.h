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

#ifndef ANAKIN_OPERATOR_PAD2D_H
#define ANAKIN_OPERATOR_PAD2D_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/pad2d.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class Pad2DHelper;

/// pad2d op
/**
* \brief Pad implementation class
* public inherit Operator
*/
template<typename Ttype, Precision Ptype>
class Pad2D : public Operator<Ttype, Ptype> {
public:
	Pad2D() {}

	/// forward impl
	virtual void operator()(OpContext<Ttype>& ctx,
	                        const std::vector<Tensor4dPtr<Ttype> >& ins,
	                        std::vector<Tensor4dPtr<Ttype> >& outs) {
				LOG(ERROR) << "Not Impl Yet Operator Pad2D< Ttype("
				           << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";
	}

	friend class Pad2DHelper<Ttype, Ptype>;
};

/**
* \brief Pad2D helper class to implement conv 3X3
* public inherit OperatorHelper
* including init resource and shape size in Permut context
*/
template<typename Ttype, Precision Ptype>
class Pad2DHelper : public OperatorHelper<Ttype, Ptype> {
public:
	Pad2DHelper() = default;

	~Pad2DHelper() {}

	Status InitParam() override;

	/**
	* \brief initial all the resource needed by pooling
	* \param ctx stand for Pad2D operation context
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
	///< _param_Pad2D stand for Pad2D parameter
	saber::Pad2DParam<Ttype> _param_pad2d;
	///< _funcs_Pad2D stand for Pad2D function
	saber::Pad2D<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_pad2d;
};

} /* namespace ops */

} /* namespace anakin */

#endif
