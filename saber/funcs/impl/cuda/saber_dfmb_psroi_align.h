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
#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DFMB_PSROI_ALIGN_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DFMB_PSROI_ALIGN_H
#include <vector>
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include "impl/impl_dfmb_psroi_algin.h"
namespace anakin {
namespace saber {
template<DataType OpDtype>
class SaberDFMBPSROIAlign<NV, OpDtype> : public ImplBase <
        NV,
        OpDtype,
    DFMBPSROIAlignParam<NV> > {
public:
    typedef Tensor<NV> OpTensor;
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberDFMBPSROIAlign() {}
    ~SaberDFMBPSROIAlign() {}
    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             DFMBPSROIAlignParam<NV>& param,
                             Context<NV>& ctx) override;

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               DFMBPSROIAlignParam<NV>& param,
                               Context<NV>& ctx);

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 DFMBPSROIAlignParam<NV>& param);
private:
    float heat_map_a_;
    float heat_map_b_;
    float pad_ratio_;

    int output_dim_;
    bool no_trans_;
    float trans_std_;
    int sample_per_part_;
    int group_height_;
    int group_width_;
    int pooled_height_;
    int pooled_width_;
    int part_height_;
    int part_width_;

    int channels_;
    int height_;
    int width_;
};

}
}
#endif