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
#ifndef ANAKIN_SABER_FUNCS_ROIS_ANCHOR_FEATURE_H
#define ANAKIN_SABER_FUNCS_ROIS_ANCHOR_FEATURE_H

#include "saber/funcs/base.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_rois_anchor_feature.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_rois_anchor_feature.h"
#endif

namespace anakin {
namespace saber {

template <typename TargetType,
        DataType OpDtype>
class RoisAnchorFeature : public BaseFunc <
        TargetType, OpDtype,
        ImplBase, RoisAnchorFeatureParam
> {
public:
    typedef TargetType targetType_t;
    typedef Tensor<TargetType> OpTensor;
    typedef RoisAnchorFeatureParam<TargetType> Param_t;
    typedef const std::vector<OpTensor*> Input_v;
    typedef std::vector<OpTensor*> Output_v;

    RoisAnchorFeature() = default;
    SaberStatus compute_output_shape(const Input_v &input,
                                     Output_v &output, Param_t &param) {

        int num_rois = input[0]->num();
        CHECK_GT(num_rois, 0);
        int rois_dim = input[0]->count(1, input[0]->dims());
        CHECK_GE(rois_dim, 5);
        int num_anchor_scales = param.num_anchor_scales;
        int num_anchors_ = num_anchor_scales * param.anchor_wph_ratios.size();
        int num_ft_per_anchor_ = param.ft_ratio_h + param.ft_ratio_w
                + param.ft_log_ratio_h + param.ft_log_ratio_w;
        int ft_dim = num_anchors_ * num_ft_per_anchor_;
        Shape output_shape({num_rois, ft_dim, 1, 1}, Layout_NCHW);

        return output[0]->set_shape(output_shape);
    }
    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderRoisAnchorFeature<TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberRoisAnchorFeature<TargetType, OpDtype>);
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