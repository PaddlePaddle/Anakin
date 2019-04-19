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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SROI_ALIGN_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SROI_ALIGN_H

#include "saber/funcs/impl/impl_sroi_align.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSRoiAlign<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        SRoiAlignParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberSRoiAlign()
    {}

    ~SaberSRoiAlign() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            SRoiAlignParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            SRoiAlignParam<ARM>& param, Context<ARM> &ctx) {
        CHECK_GT(param.pooled_h, 0) << "pooled_h must be > 0";
        CHECK_GT(param.pooled_w, 0) << "pooled_w must be > 0";

        _pooled_height = param.pooled_h;
        _pooled_width = param.pooled_w;
        _spatial_scale = param.spatial_scale;

        _channels = inputs[0]->channel();
        _height = inputs[0]->height();
        _width = inputs[0]->width();
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          SRoiAlignParam<ARM>& param);
private:
    Tensor<ARM> _tmp_in;
    int _channels;
    int _height;
    int _width;
    int _pooled_height;
    int _pooled_width;
    float _spatial_scale;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SRoiAlign_H
