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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_DECONV_ACT_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_DECONV_ACT_H

#include "saber/lite/funcs/saber_deconv.h"

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberDeconvAct2D : public OpBase {
public:
    SaberDeconvAct2D() {
        _conv_func = new SaberDeconv2D;
    }

    SaberDeconvAct2D(const ParamBase* param) {
        _conv_func = new SaberDeconv2D;
        _param = (const ConvAct2DParam*)param;
       /*
        if (_param->_flag_act) {
            LCHECK_EQ(_param->_act_type, Active_relu, "active type must be relu");
        }
        */
        this->_flag_param = true;
        _conv_func->load_param(&_param->_conv_param);
    }

    virtual SaberStatus load_param(const ParamBase* param) override {
        _param = (const ConvAct2DParam*)param;
        this->_flag_param = true;
        _conv_func->set_activation(_param->_flag_act);
        return _conv_func->load_param(&_param->_conv_param);
    }


//    SaberConvAct2D(int weights_size, int num_output, int group, int kw, int kh, int stride_w, int stride_h, \
//        int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, ActiveType type, bool flag_relu, \
//        const float* weights, const float* bias) {
//
//        if (flag_relu) {
//            LCHECK_EQ(type, Active_relu, "active type must be relu");
//            _flag_relu = true;
//        } else {
//            _flag_relu = false;
//        }
//        SaberDeconv2D::set_activation(_flag_relu);
//        SaberDeconv2D::load_param(weights_size, num_output, group, kw, kh, stride_w, stride_h, \
//            pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias);
//    }
//
//    SaberStatus load_param(int weights_size, int num_output, int group, int kw, int kh, int stride_w, int stride_h, \
//        int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, ActiveType type, bool flag_relu, \
//        const float* weights, const float* bias) {
//
//        if (flag_relu) {
//            LCHECK_EQ(type, Active_relu, "active type must be relu");
//            _flag_relu = true;
//        } else {
//            _flag_relu = false;
//        }
//        SaberDeconv2D::set_activation(_flag_relu);
//        return SaberDeconv2D::load_param(weights_size, num_output, group, kw, kh, stride_w, stride_h, \
//            pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias);
//    }

    ~SaberDeconvAct2D() {
        delete _conv_func;
    }

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs)  override{
        if (!this->_flag_param) {
            printf("load conv_act param first\n");
            return SaberNotInitialized;
        }
        return _conv_func->compute_output_shape(inputs, outputs);
    }

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override{
        if (!this->_flag_param) {
            printf("load conv_act param first\n");
            return SaberNotInitialized;
        }
        if (_param->_flag_act) {
            _conv_func->set_activation(true);
            //SABER_CHECK(_conv_func->set_activation(true));
        } else {
            _conv_func->set_activation(false);
           // SABER_CHECK(_conv_func->set_activation(false));
        }
        // LOG(INFO) << "Deconv act";
        //_conv_func->set_activation(_param->_flag_act);
        this->_flag_init = true;
        return _conv_func->init(inputs, outputs, ctx);
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) override{
        if (!this->_flag_init) {
            printf("init conv_act first\n");
            return SaberNotInitialized;
        }
        return _conv_func->dispatch(inputs, outputs);
    }
private:
    const ConvAct2DParam* _param;
    SaberDeconv2D* _conv_func;
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H
