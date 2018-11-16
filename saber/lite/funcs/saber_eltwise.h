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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_ELTWISE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_ELTWISE_H

#include "saber/lite/funcs/op_base.h"
#include "saber/lite/funcs/calibrate_lite.h"
#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

typedef void (*eltwise_func)(const void* din_a, \
    const void* din_b, void* dout, const int size, std::vector<float> coef);

typedef signed char char_t;
//template <typename Dtype>
class SaberEltwise : public OpBase {
public:
    SaberEltwise() {}

    SaberEltwise(ParamBase* param);

    virtual SaberStatus load_param(ParamBase* param) override;

    virtual SaberStatus set_op_precision(DataType ptype) override;

    virtual SaberStatus load_param(std::istream& stream, const float* weights) override;

    ~SaberEltwise();

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU>*>& inputs,
                                     std::vector<Tensor<CPU>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU>*>& inputs,
                             std::vector<Tensor<CPU>*>& outputs, Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU>*>& inputs, \
                                 std::vector<Tensor<CPU>*>& outputs) override;

private:
    EltwiseParam* _param;
    eltwise_func _impl{nullptr};

    std::vector<Tensor<CPU>> _tmp_in;
    Tensor<CPU> _tmp_out;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_ELTWISE_H
