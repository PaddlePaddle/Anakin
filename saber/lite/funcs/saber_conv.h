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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_CONV_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_CONV_H

#include "saber/lite/funcs/op_base.h"
#include "saber/lite/funcs/saber_activation.h"
#include "saber/lite/funcs/calibrate_lite.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

typedef void (*conv_func)(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr);


//template <typename Dtype>
class SaberConv2D : public OpBase {
public:
    SaberConv2D();

    SaberConv2D(ParamBase* param);

    virtual SaberStatus load_param(ParamBase* param) override;

    virtual SaberStatus set_op_precision(DataType ptype) override;

    virtual SaberStatus load_param(std::istream& stream, const float* weights) override;

    ~SaberConv2D();

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU>*>& inputs,
                                     std::vector<Tensor<CPU>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU>*>& inputs,
                             std::vector<Tensor<CPU>*>& outputs, Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU>*>& inputs,
                                 std::vector<Tensor<CPU>*>& outputs) override;

private:
    Conv2DParam* _param;
    conv_func _impl{nullptr};
    bool _is_trans_weights{false};
    bool _is_int8_weights{false};

    std::vector<float> _w_scale;
    Tensor<CPU> _weights_trans;
    Tensor<CPU> _idx_data;

    Tensor<CPU> _tmp_in;
    Tensor<CPU> _tmp_out;

    SaberActivation* _act_funcs{nullptr};
    ActivationParam* _act_param{nullptr};
#ifdef ENABLE_OP_TIMER
    std::string _conv_type;
#endif
};


} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_H
