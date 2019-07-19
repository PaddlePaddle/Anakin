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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ROI_ALIGN_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ROI_ALIGN_H

#include "saber/funcs/impl/impl_roi_align.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberRoiAlign<NV, OpDtype>:
    public ImplBase<NV, OpDtype, RoiAlignParam<NV>> {

public:

    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberRoiAlign()
    {}

    ~SaberRoiAlign() {

    }

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             RoiAlignParam<NV> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;
        create(inputs, outputs, param, ctx);

        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               RoiAlignParam<NV> &param,
                               Context<NV> &ctx) {

        Shape out_stride = outputs[0]->get_stride();
        Shape in_stride = inputs[0]->get_stride();
        _in_n_stride = in_stride[0];
        _in_c_stride = in_stride[1];
        _in_h_stride = in_stride[2];
        _in_w_stride = in_stride[3];
        _out_n_stride = out_stride[0];
        _out_c_stride = out_stride[1];
        _out_h_stride = out_stride[2];
        _out_w_stride = out_stride[3];

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 RoiAlignParam<NV> &param);

private:
    int _in_n_stride;
    int _in_c_stride;
    int _in_h_stride;
    int _in_w_stride;
    int _out_n_stride;
    int _out_c_stride;
    int _out_h_stride;
    int _out_w_stride;
    const int _kROISize = 5;
};

}

}

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ROI_POOL_H
