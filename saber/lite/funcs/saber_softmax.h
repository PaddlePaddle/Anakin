/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_SOFTMAX_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_SOFTMAX_H
#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
class SaberSoftmax{
public:

    SaberSoftmax() = default;
    ~SaberSoftmax() {}


    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                              std::vector<Tensor<Dtype>*>& outputs,
                              SoftmaxParam<Tensor<Dtype>> &param) {
        return outputs[0]->set_shape(inputs[0]->valid_shape());
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs,
                             std::vector<Tensor<Dtype>*>& outputs,
                             SoftmaxParam<Tensor<Dtype>> &param, Context &ctx) {
        // get context
        _ctx = ctx;
        return create(inputs, outputs, param, ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs,
                               std::vector<Tensor<Dtype>*>& outputs,
                               SoftmaxParam<Tensor<Dtype>> &param, Context &ctx) {

        Shape shape_in = inputs[0]->valid_shape();
        Shape shape_out = outputs[0]->valid_shape();
        _outer_num = inputs[0]->count_valid(0, param.axis);
        _inner_num = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
        _axis_size = shape_in[param.axis];

        int buffer_size = this->_inner_num * this->_outer_num;
        return SaberSuccess;
    }

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs,
                                 std::vector<Tensor<Dtype>*>& outputs,
                                 SoftmaxParam<Tensor<Dtype>> &param);

private:
    Context _ctx;
    int _axis_size{0};
    int _inner_num{0};
    int _outer_num{0};

};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_SOFTMAX_H
