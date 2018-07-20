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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_PRIORBOX_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_PRIORBOX_H

#include "saber/lite/funcs/op_base.h"

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberPriorBox : public OpBase {
public:

    SaberPriorBox() = default;

    SaberPriorBox(const ParamBase* param);

    virtual SaberStatus load_param(const ParamBase* param) override;

//    SaberPriorBox(bool is_flip, bool is_clip, std::vector<float> min_size, std::vector<float> max_size, \
//        std::vector<float> aspect_ratio, std::vector<float> variance, \
//        int img_width, int img_height, float step_w, float step_h, float offset);
//
//    SaberStatus load_param(bool is_flip, bool is_clip, std::vector<float> min_size, std::vector<float> max_size, \
//        std::vector<float> aspect_ratio, std::vector<float> variance, \
//        int img_width, int img_height, float step_w, float step_h, float offset);

    ~SaberPriorBox() {}

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                        std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                          std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

private:
    const PriorBoxParam* _param;
    Tensor<CPU, AK_FLOAT> _output_arm;
    int _num_priors;

//    bool _is_flip;
//    bool _is_clip;
//    std::vector<float> _min_size;
//    std::vector<float> _max_size;
//    std::vector<float> _aspect_ratio;
//    std::vector<float> _variance;
//    int _img_width;
//    int _img_height;
//    float _step_w;
//    float _step_h;
//    float _offset;
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_PRIORBOX_H
