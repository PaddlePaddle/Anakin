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
#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ROIS_ANCHOR_FEATURE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ROIS_ANCHOR_FEATURE_H

#include <vector>
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_rois_anchor_feature.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberRoisAnchorFeature<NV, OpDtype> : public ImplBase <
        NV, OpDtype, RoisAnchorFeatureParam<NV> > {
public:

    SaberRoisAnchorFeature() {}
    ~SaberRoisAnchorFeature() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>*> &inputs,
                             std::vector<Tensor<NV>*> &outputs,
                             RoisAnchorFeatureParam<NV>& param,
                             Context<NV>& ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<NV>*> &inputs,
                               std::vector<Tensor<NV>*> &outputs,
                               RoisAnchorFeatureParam<NV>& param,
                               Context<NV>& ctx) override;

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*> &inputs,
                                 std::vector<Tensor<NV>*> &outputs,
                                 RoisAnchorFeatureParam<NV>& param) override;
private:
    bool _has_inited{false};
    int num_anchors_;
    int num_top_iou_anchor_;
    int min_num_top_iou_anchor_;
    float iou_thr_;
    std::vector<float> anchor_width_;
    std::vector<float> anchor_height_;
    std::vector<float> anchor_area_;
    bool ft_ratio_h_;
    bool ft_ratio_w_;
    bool ft_log_ratio_h_;
    bool ft_log_ratio_w_;
    int num_ft_per_anchor_;
    bool bbox_size_add_one_;
    Tensor<NVHX86> bottom;
    Tensor<NVHX86> top;
};
}
}
#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
