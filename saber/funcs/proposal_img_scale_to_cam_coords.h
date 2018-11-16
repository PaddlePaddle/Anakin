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
#ifndef ANAKIN_SABER_FUNCS_PROPOSAL_IMG_SCALE_TO_CAM_COORDS_H
#define ANAKIN_SABER_FUNCS_PROPOSAL_IMG_SCALE_TO_CAM_COORDS_H
#include "saber/funcs/base.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_proposal_img_scale_to_cam_coords.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_proposal_img_scale_to_cam_coords.h"
#endif

namespace anakin {
namespace saber {

template <typename TargetType,
        DataType OpDtype>
class ProposalImgScaleToCamCoords : public BaseFunc <
        TargetType, OpDtype,
        ImplBase, ProposalImgScaleToCamCoordsParam
> {
public:
    typedef TargetType targetType_t;
    typedef Tensor<TargetType> OpTensor;
    typedef ProposalImgScaleToCamCoordsParam<TargetType> Param_t;
    typedef const std::vector<OpTensor*> Input_v;
    typedef std::vector<OpTensor*> Output_v;

    ProposalImgScaleToCamCoords() = default;
    SaberStatus compute_output_shape(const Input_v &input,
                                     Output_v &output, Param_t &param) {
        int num_top_channels_ = 3;

        if (param.has_size3d_and_orien3d) {

            num_top_channels_ += 4;
        }

        if (param.cmp_pts_corner_3d) {
            CHECK(param.has_size3d_and_orien3d);
            num_top_channels_ += 24;
        }

        if (param.cmp_pts_corner_2d) {
            CHECK(param.cmp_pts_corner_3d);
            num_top_channels_ += 16;
        }

        Shape output_shape({1, num_top_channels_, 1, 1}, Layout_NCHW);
        //    outputs[0]->Reshape(1, num_top_channels_, 1, 1);
        return output[0]->set_shape(output_shape);
    }
    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderProposalImgScaleToCamCoords<TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberProposalImgScaleToCamCoords<TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }

    };
private:
    virtual void pick_best_static() override {
        if (true) { // some condition?
            this->_best_impl = this->_impl[0];
        }
    }
    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }
};
}
}
#endif //ANAKIN_SABER_FUNCS_CONV_H