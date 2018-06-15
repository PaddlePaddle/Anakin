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

#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_ACTIVATION_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_ACTIVATION_H

#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
class SaberActivation {
public:
    SaberActivation() {}

    ~SaberActivation() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     ActivationParam<Tensor<Dtype>> &param) {
        return outputs[0]->set_shape(inputs[0]->valid_shape());
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs,
                            std::vector<Tensor<Dtype>*>& outputs,
                            ActivationParam<Tensor<Dtype>>& param, Context& ctx) {
        _ctx = ctx;
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs,
                            std::vector<Tensor<Dtype>*>& outputs,
                            ActivationParam<Tensor<Dtype>>& param, Context &ctx) {
        _ctx = ctx;
        return SaberSuccess;
    }
    
    virtual SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs,
                          std::vector<Tensor<Dtype>*>& outputs,
                          ActivationParam<Tensor<Dtype>>& param) {
        const Dtype* din = inputs[0]->data();
        Dtype* dout = outputs[0]->mutable_data();
        int size = outputs[0]->valid_size();
        if (param.active == Active_relu) {
            for (int i = 0; i < size; ++i) {
                dout[i] = std::max(din[i], (Dtype)0);
            }
            return SaberSuccess;
        } else {
            return SaberUnImplError;
        }
    }
private:
    Context _ctx;
};


} //namespace lite

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_LITE_FUNCS_SABER_ACTIVATION_H
