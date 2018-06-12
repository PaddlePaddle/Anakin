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

#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_CONCAT_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_CONCAT_H

#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
class SaberConcat {
public:
    SaberConcat() = default;
    ~SaberConcat() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     ConcatParam<Tensor<Dtype>> &param) {
        unsigned long input_size = inputs.size();

        Shape shape_out = inputs[0]->valid_shape();

        //! compute output shape
        for (int i = 1; i < input_size; ++i) {
            Shape sh = inputs[i]->valid_shape();
            for (int j = 0; j < sh.dims(); ++j) {
                if (j == param.axis) { continue; }

                CHECK_EQ(shape_out[j], sh[j]) \
                        << "All inputs must have the same shape, except at concat_axis.";
            }
            shape_out[param.axis] += sh[param.axis];
        }
        return outputs[0]->set_shape(shape_out);
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs,
                      std::vector<Tensor<Dtype>*>& outputs,
                      ConcatParam<Tensor<Dtype>> &param, Context &ctx){
        return create(inputs, outputs, param, ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs,
                        std::vector<Tensor<Dtype>*>& outputs,
                        ConcatParam<Tensor<Dtype>> &param, Context &ctx){
        _ctx = ctx;
        _num_concats = inputs[0]->count_valid(0, param.axis);
        _concat_input_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
        return SaberSuccess;
    }

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs,
                          std::vector<Tensor<Dtype>*>& outputs,
                          ConcatParam<Tensor<Dtype>> &param);

private:
    Context _ctx;
    int _num_concats;
    int _concat_input_size;
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONCAT_H
