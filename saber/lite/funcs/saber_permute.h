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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_PERMUTE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_PERMUTE_H

#include "saber/lite/funcs/op_base.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberPermute : public OpBase {
public:
    SaberPermute();

    SaberPermute(ParamBase* param);

    ~SaberPermute();

    virtual SaberStatus load_param(ParamBase* param) override;

    virtual SaberStatus set_op_precision(DataType ptype) override;

    virtual SaberStatus load_param(std::istream& stream, const float* weights) override;

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU>*>& inputs,
                                     std::vector<Tensor<CPU>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU>*>& inputs, \
        std::vector<Tensor<CPU>*>& outputs, Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU>*>& inputs, \
        std::vector<Tensor<CPU>*>& outputs) override;


private:
    PermuteParam* _param;
    int _num_axes;
    int _count;
    bool _need_permute{false};
    bool _transpose{false};
    int _trans_num;
    int _trans_w;
    int _trans_h;
    std::vector<int> _new_steps;
    std::vector<int> _old_steps;

};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_PERMUTE_H
