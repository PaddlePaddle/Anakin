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

#ifndef ANAKIN_OPERATOR_PROPOSAL_IMG_SCALE_TO_CAM_COORDS_H
#define ANAKIN_OPERATOR_PROPOSAL_IMG_SCALE_TO_CAM_COORDS_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/proposal_img_scale_to_cam_coords.h"

namespace anakin {
namespace ops {
template<typename Ttype, Precision Ptype>
class ProposalImgScaleToCamCoordsHelper;
/// pooling op
/**
* \brief ProposalImgScaleToCamCoords implementation class
* public inherit Operator
*/
template<typename Ttype, Precision Ptype>
class ProposalImgScaleToCamCoords : public Operator<Ttype, Ptype> {
public:
    ProposalImgScaleToCamCoords() {}
    /// forward impl
    virtual void operator()(OpContext<Ttype>& ctx,
                            const std::vector<Tensor4dPtr<Ttype> >& ins,
                            std::vector<Tensor4dPtr<Ttype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator Proposal_img_scale_to_cam_coords<TargetType:" <<
                   target_name<Ttype>::value << "), Precision(" << Ptype << ") >";
    }
    friend class ProposalImgScaleToCamCoordsHelper<Ttype, Ptype>;
};
/**
* \brief ProposalImgScaleToCamCoords helper class to implement ProposalImgScaleToCamCoords
* public inherit OperatorHelper
* including init resource and shape size in ProposalImgScaleToCamCoords context
*/
template<typename Ttype, Precision Ptype>
class ProposalImgScaleToCamCoordsHelper : public OperatorHelper<Ttype, Ptype> {
public:
    ProposalImgScaleToCamCoordsHelper() = default;
    ~ProposalImgScaleToCamCoordsHelper();
    Status InitParam() override;
    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for ProposalImgScaleToCamCoords operation context
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
    ///< _param_proposal_img_scale_to_cam_coords stand for ProposalImgScaleToCamCoords parameter
    saber::ProposalImgScaleToCamCoordsParam<Ttype>  _param_proposal_img_scale_to_cam_coords;
    ///< _funcs_proposal_img_scale_to_cam_coords stand for ProposalImgScaleToCamCoords function
    saber::ProposalImgScaleToCamCoords<Ttype,
        PrecisionWrapper<Ptype>::saber_type> _funcs_proposal_img_scale_to_cam_coords;
};
} /* namespace ops */
} /* namespace anakin */
#endif