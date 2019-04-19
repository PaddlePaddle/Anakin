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
#ifndef ANAKIN_OPERATOR_ROI_POOLING_H
#define ANAKIN_OPERATOR_ROI_POOLING_H
#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/roi_pooling.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class RoiPoolHelper;

/**
* \brief RoiPool implementation class
* public inherit Operator
*/
template<typename Ttype, Precision Ptype>
class RoiPool : public Operator<Ttype, Ptype> {
public:
    RoiPool() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx,
                             const std::vector<Tensor4dPtr<Ttype> >& ins,
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
                LOG(ERROR) << "Not Impl Yet Operator RoiPooling< Ttype("
                           << target_name<Ttype>::value << "), Precision(" << Ptype << ") >";
    }

    friend class RoiPoolHelper<Ttype, Ptype>;
};

/**
* \brief RoiPool helper class to implement RoiPool
* public inherit OperatorHelper
* including init resource and shape size in RoiPool context
*/
template<typename Ttype, Precision Ptype>
class RoiPoolHelper : public OperatorHelper<Ttype, Ptype> {
public:
    RoiPoolHelper()=default;

    ~RoiPoolHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for RoiPool operation context
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
    ///< _param_roi_pool stand for RoiPool parameter
    saber::RoiPoolParam<Ttype> _param_roi_pool;
    ///< _funcs_roi_pool stand for RoiPool function
    saber::RoiPool<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs_roi_pool;

};

} /* namespace ops */

} /* namespace anakin */
#endif //ANAKIN_OPERATOR_ROI_POOLING_H
