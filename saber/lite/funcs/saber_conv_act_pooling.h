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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_POOLING_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_POOLING_H

#include "saber/lite/funcs/saber_conv_act.h"
#include "saber/lite/funcs/saber_pooling.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberConvActPooling2D : public OpBase {
public:
    SaberConvActPooling2D();

    SaberConvActPooling2D(const ParamBase* param);

    virtual SaberStatus load_param(const ParamBase* param) override;

    virtual SaberStatus load_param(FILE* fp, const float* weights) override;

    ~SaberConvActPooling2D();

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) override;

private:
    //bool _flag_relu;
    const ConvActPool2DParam* _param;
    Tensor<CPU, AK_FLOAT> _tensor_tmp;
    std::vector<Tensor<CPU, AK_FLOAT> *> _vtensor_tmp;
    SaberConvAct2D* _conv_act_func;
    SaberPooling* _pool_func;
};
} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_POOLING_H
