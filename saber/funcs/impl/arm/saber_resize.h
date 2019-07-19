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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_RESIZE_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_RESIZE_H

#include "saber/funcs/impl/impl_resize.h"

namespace anakin{

namespace saber{

typedef void (*resize_func)(const float* src, int w_in, int h_in, float* dst, \
              int w_out, int h_out, float scale_x, float scale_y, ResizeType resize_type);

template <DataType OpDtype>
class SaberResize<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        ResizeParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberResize()
    {}

    ~SaberResize() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            ResizeParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            ResizeParam<ARM>& param, Context<ARM> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          ResizeParam<ARM>& param);
private:
  resize_func _impl{nullptr};
  float _width_scale{0.0f};
  float _height_scale{0.0f};
  Shape _src_real_shape;
  Shape _dst_real_shape;
  ResizeType _resize_type;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Resize_H
