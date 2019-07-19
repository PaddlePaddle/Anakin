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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_PAD2D_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_PAD2D_H

#include "saber/funcs/impl/impl_pad2d.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPad2D<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        Pad2DParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberPad2D()
    {}

    ~SaberPad2D() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            Pad2DParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            Pad2DParam<ARM>& param, Context<ARM> &ctx) {
      _mode = param._mode;
      _pad_h = param._pad_h;
      _pad_w = param._pad_w;
      _pad_value = param._pad_value;
      if (_mode == PAD_REFLECT){
        CHECK_LE(_pad_h[0], inputs[0]->height() - 1) << "pad top size must <= inputs height - 1";
        CHECK_LE(_pad_h[1], inputs[0]->height() - 1) << "pad bottom size must <= inputs height - 1";
        CHECK_LE(_pad_w[0], inputs[0]->width() - 1) << "pad left size must <= inputs width - 1";
        CHECK_LE(_pad_w[1], inputs[0]->width() - 1) << "pad right size must  <= inputs width - 1";
      }
      return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          Pad2DParam<ARM>& param);
private:
    PadMode _mode;
    std::vector<int> _pad_h{0, 0};
    std::vector<int> _pad_w{0, 0};
    float _pad_value = 0.f;

};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Pad2D_H
