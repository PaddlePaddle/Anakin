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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PS_ROI_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PS_ROI_POOLING_H

#include "saber/funcs/impl/impl_ps_roi_pooling.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPsRoiPool<NV, OpDtype>:
    public ImplBase<NV, OpDtype, PsRoiPoolParam<NV>> {

public:

    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberPsRoiPool()
    {}

    ~SaberPsRoiPool() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             PsRoiPoolParam<NV> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;
        
        
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               PsRoiPoolParam<NV> &param,
                               Context<NV> &ctx) {
        Shape inter_shape = inputs[0]->shape();
        int oc = outputs[0]->channel();
        int num = outputs[0]->num();
        inter_shape.set_num(param.pooled_height * param.pooled_width * oc);
        inter_shape.set_channel(num);
        inter_shape.set_width(param.crop_height);
        inter_shape.set_height(param.crop_width);
        _crop_data.re_alloc(inter_shape, OpDtype);
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 PsRoiPoolParam<NV> &param);

private:
  Tensor<NV> _crop_data;

    
};
template class SaberPsRoiPool<NV, AK_FLOAT>;
}

}

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ROI_POOL_H
