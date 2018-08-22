/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_OPERATOR_PRIORBOX_H
#define ANAKIN_OPERATOR_PRIORBOX_H

#include "framework/core/operator/operator.h"
#include "saber/funcs/priorbox.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class PriorBoxHelper;

//! PriorBox op
template<typename Ttype, Precision Ptype>
class PriorBox : public Operator<Ttype, Ptype> {
public:
    PriorBox() {}

    //! forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
		LOG(ERROR) << "Not Impl Yet Operator PriorBox< Ttype("
				   << target_name<Ttype>::value << "), Precision("<< Ptype <<") >";	
    }

    friend class PriorBoxHelper<Ttype, Ptype>;
};

template<typename Ttype, Precision Ptype>
class PriorBoxHelper : public OperatorHelper<Ttype, Ptype> {
public:
    PriorBoxHelper()=default;

    ~PriorBoxHelper() {}

    Status InitParam() override;

    //! initial all the resource needed by pooling
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype> >& ins, 
                std::vector<Tensor4dPtr<Ttype> >& outs) override;

    //! infer the shape of output and input.
    Status InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                      std::vector<Tensor4dPtr<Ttype> >& outs) override;

public:
    saber::PriorBoxParam<Ttype> _param_priorbox;
    saber::PriorBox<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_priorbox;
};

} /* namespace ops */

} /* namespace anakin */

#endif //ANAKIN_OPERATOR_PRIORBOX_H
