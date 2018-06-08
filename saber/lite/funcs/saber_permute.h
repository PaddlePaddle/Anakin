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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_PERMUTE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_PERMUTE_H

#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
class SaberPermute {
public:
    SaberPermute() {
        _transpose = false;
        _need_permute = false;
    }
    ~SaberPermute() {}


    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     PermuteParam<Tensor<Dtype>> &param) {
        SaberStatus status;
        for (int i = 0; i < inputs.size(); ++i) {
            Shape output_shape = inputs[i]->valid_shape();

            if (inputs[i]->valid_shape().size() != param.order.size()) {
                LOG(FATAL) << "permute order param is not valid";
            }

            //for example: (n, h, w, c)->(n, c, h, w)  by order(0, 3, 1, 2)
            for (int j = 0; j < param.order.size(); j++) {
                output_shape[j] = inputs[i]->valid_shape()[param.order[j]];
            }
            outputs[i]->set_shape(output_shape);
        }
        return SaberSuccess;
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs, \
        std::vector<Tensor<Dtype>*>& outputs, \
        PermuteParam<Tensor<Dtype>> &param, Context &ctx) {
        return create(inputs, outputs, param, ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs, \
        std::vector<Tensor<Dtype>*>& outputs, \
        PermuteParam<Tensor<Dtype>> &param, Context &ctx);

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs, \
        std::vector<Tensor<Dtype>*>& outputs, PermuteParam<Tensor<Dtype>> &param);


private:
    Context _ctx;
    int _num_axes;
    int _count;
    bool _need_permute{false};
    bool _transpose{false};
    int _trans_num;
    int _trans_w;
    int _trans_h;
    std::vector<int> _order_dims;
    std::vector<int> _new_steps;
    std::vector<int> _old_steps;

};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_PERMUTE_H
