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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_POOLING_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_POOLING_H

#include "saber/lite/funcs/op_base.h"
#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

typedef void (*pool_func)(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          PoolingType type, bool global, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int pad_w, int pad_h);

//template <typename Dtype>
class SaberPooling : public OpBase {

public:
    SaberPooling() {}

    SaberPooling(const ParamBase* param);

    virtual SaberStatus load_param(const ParamBase* param) override;

    //virtual SaberStatus load_param(FILE* fp, const float* weights) override;

    virtual SaberStatus load_param(std::istream& stream, const float* weights) override;

    ~SaberPooling();

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
        std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
        std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

private:
    const PoolParam* _param;
    pool_func _impl{nullptr};
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_POOLING_H
