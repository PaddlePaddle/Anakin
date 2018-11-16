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

#include "saber/lite/funcs/op_base.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberSoftmax : public OpBase {
public:

    SaberSoftmax() = default;

    SaberSoftmax(ParamBase* param);

    virtual SaberStatus load_param(ParamBase* param) override;

    virtual SaberStatus set_op_precision(DataType ptype) override;

    virtual SaberStatus load_param(std::istream& stream, const float* weights) override;

    ~SaberSoftmax();


    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU>*>& inputs,
                              std::vector<Tensor<CPU>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU>*>& inputs,
                               std::vector<Tensor<CPU>*>& outputs, Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU>*>& inputs,
                                 std::vector<Tensor<CPU>*>& outputs) override;

private:
    SoftmaxParam* _param;
    int _axis_size{0};
    int _inner_num{0};
    int _outer_num{0};
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_SOFTMAX_H
