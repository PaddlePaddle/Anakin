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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_ROI_ALIGN_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_ROI_ALIGN_H

#include "saber/funcs/impl/impl_roi_align.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberRoiAlign<X86, OpDtype>:
    public ImplBase<X86, OpDtype, RoiAlignParam<X86>> {

public:

    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberRoiAlign()
    {}

    ~SaberRoiAlign() {

    }

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             RoiAlignParam<X86> &param,
                             Context<X86> &ctx) {
        this->_ctx = &ctx;
        Shape out_stride = outputs[0]->get_stride();
        Shape in_stride = inputs[0]->get_stride();
        int in_n_index = inputs[0]->num_index();
        int in_c_index = inputs[0]->channel_index();
        int in_h_index = inputs[0]->height_index();
        int in_w_index = inputs[0]->width_index();
        int out_n_index = outputs[0]->num_index();
        int out_c_index = outputs[0]->channel_index();
        int out_h_index = outputs[0]->height_index();
        int out_w_index = outputs[0]->width_index();
        _in_n_stride = in_stride[in_n_index];
        _in_c_stride = in_stride[in_c_index];
        _in_h_stride = in_stride[in_h_index];
        _in_w_stride = in_stride[in_w_index];
        _out_n_stride = out_stride[out_n_index];
        _out_c_stride = out_stride[out_c_index];
        _out_h_stride = out_stride[out_h_index];
        _out_w_stride = out_stride[out_w_index];
        
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               RoiAlignParam<X86> &param,
                               Context<X86> &ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 RoiAlignParam<X86> &param);

private:
    int _in_n_stride;
    int _in_c_stride;
    int _in_h_stride;
    int _in_w_stride;
    int _out_n_stride;
    int _out_c_stride;
    int _out_h_stride;
    int _out_w_stride;
    const int _prePosROISize = 4;
    const int _kROISize = 5;
    Tensor<X86> _pre_pos;
    Tensor<X86> _pre_w;
};

}

}

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_ROI_POOL_H