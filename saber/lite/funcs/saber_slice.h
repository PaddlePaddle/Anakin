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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_SLICE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_SLICE_H

#include "saber/lite/funcs/op_base.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{
//template <typename Dtype>
class SaberSlice : public OpBase {
public:

    SaberSlice() {
        _slice_num = 4;
        _slice_size = 0;
    }

    SaberSlice(const ParamBase* param);

    virtual SaberStatus load_param(const ParamBase* param) override;

    virtual SaberStatus load_param(FILE* fp, const float* weights) override;

    ~SaberSlice();

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

private:
    const SliceParam* _param;
    int _slice_num;
    int _slice_size;

    std::vector<int> _slice_points;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_SLICE_H
