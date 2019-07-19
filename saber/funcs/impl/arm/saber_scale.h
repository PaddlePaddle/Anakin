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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SCALE_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SCALE_H

#include "saber/funcs/impl/impl_scale.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberScale<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        ScaleParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberScale()
    {}

    ~SaberScale() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            ScaleParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            ScaleParam<ARM>& param, Context<ARM> &ctx) {
        const int count = inputs[0]->valid_size();
        int axis = (param.num_axes == 0) ? 0 : param.axis;
        int num_axes = param.num_axes >=0 ? param.num_axes : inputs[0]->shape().dims() - axis;
        CHECK_LE(axis + num_axes, inputs[0]->shape().dims());
        _inner_dim = inputs[0]->count(axis + num_axes, inputs[0]->shape().dims());
        _scale_dim = inputs[0]->count(axis, axis + num_axes);
        _outer_dim = inputs[0]->count(0, axis);
        if (inputs.size() > 1) {
            CHECK_EQ(_scale_dim, inputs[1]->valid_size()) << "scale dim not valid";
        } else {
            CHECK_EQ(_scale_dim, param.scale_w.size()) << "scale dim not valid";
        }
        // _inner_dim = inputs[0]->count(param.axis + param.num_axes, inputs[0]->shape().dims());
        // _scale_dim = inputs[0]->count(param.axis, param.axis + param.num_axes);

        // const int count = inputs[0]->valid_size();

        // if (inputs.size() > 1) {
        //     _scale_dim = inputs[1]->valid_size();
        //     _inner_dim = count / _scale_dim;
        // }
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          ScaleParam<ARM>& param);
private:
  int _scale_dim;
  int _inner_dim;
  int _outer_dim;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Scale_H
