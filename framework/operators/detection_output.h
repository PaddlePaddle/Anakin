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

#ifndef ANAKIN_OPERATOR_DETECTION_OUTPUT_H
#define ANAKIN_OPERATOR_DETECTION_OUTPUT_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/detection_output.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class DetectionOutputHelper;

//! DetectionOutput op
template<typename Ttype, DataType Dtype, Precision Ptype>
class DetectionOutput : public Operator<Ttype, Dtype, Ptype> {
public:
    DetectionOutput() {}

    //! forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        //LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
         //          <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class DetectionOutputHelper<Ttype, Dtype, Ptype>;
};

template<typename Ttype, DataType Dtype, Precision Ptype>
class DetectionOutputHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    DetectionOutputHelper()=default;

    ~DetectionOutputHelper();

    Status InitParam() override;

    //! initial all the resource needed by pooling
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

    //! infer the shape of output and input.
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override;

public:
    saber::DetectionOutputParam<Tensor4d<Ttype, Dtype>> _param_detection_output;
    saber::DetectionOutput<Ttype, Dtype> _funcs_detection_output;
};



} /* namespace ops */

} /* namespace anakin */

#endif //ANAKIN_OPERATOR_DETECTION_OUTPUT_H
