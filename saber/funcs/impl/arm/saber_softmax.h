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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SOFTMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SOFTMAX_H

#include "saber/funcs/impl/impl_softmax.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSoftmax<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        SoftmaxParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberSoftmax()
    {}

    ~SaberSoftmax() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            SoftmaxParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            SoftmaxParam<ARM>& param, Context<ARM> &ctx) {
      Shape shape_in = inputs[0]->valid_shape();
      Shape shape_out = outputs[0]->valid_shape();
      _outer_num = inputs[0]->count_valid(0, param.axis);
      _inner_num = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
      _axis_size = shape_in[param.axis];
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          SoftmaxParam<ARM>& param);
private:
    int _axis_size{0};
    int _inner_num{0};
    int _outer_num{0};
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Softmax_H
