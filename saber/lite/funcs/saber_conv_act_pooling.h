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
    SaberConvActPooling2D() {
        _pool_func = new SaberPooling;
        _conv_act_func = new SaberConvAct2D;
        _vtensor_tmp.push_back(&_tensor_tmp);
    }

    SaberConvActPooling2D(const ParamBase* param) {
        _pool_func = new SaberPooling;
        _conv_act_func = new SaberConvAct2D;
        _param = (const ConvActPool2DParam*)param;
        _conv_act_func->load_param(&_param->_conv_act_param);
        _pool_func->load_param(&_param->_pool_param);
        this->_flag_param = true;
    }

    virtual SaberStatus load_param(const ParamBase* param) override {

        _param = (const ConvActPool2DParam*)param;
        this->_flag_param = true;
        SaberStatus state = _conv_act_func->load_param(&_param->_conv_act_param);
        if (state != SaberSuccess) {
            printf("load conv2d failed\n");
            return state;
        }
        return _pool_func->load_param(&_param->_pool_param);
    }

    ~SaberConvActPooling2D() {
        delete _pool_func;
        delete _conv_act_func;
    }

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override {

        if (!this->_flag_param) {
            printf("load conv_act_pool param first\n");
            return SaberNotInitialized;
        }

        SaberStatus state = _conv_act_func->compute_output_shape(inputs, _vtensor_tmp);
        if (state != SaberSuccess) {
            return state;
        }
        _tensor_tmp.re_alloc(_tensor_tmp.valid_shape());
        return _pool_func->compute_output_shape(_vtensor_tmp, outputs);
    }

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override {

        if (!this->_flag_param) {
            printf("load conv_act_pool param first\n");
            return SaberNotInitialized;
        }

        //SaberConv2D::set_activation(_param->_flag_act);
        _tensor_tmp.re_alloc(_tensor_tmp.valid_shape());
        this->_flag_init = true;
        SaberStatus state = _conv_act_func->init(inputs, _vtensor_tmp, ctx);
        if (state != SaberSuccess) {
            return state;
        }
        return _pool_func->init(_vtensor_tmp, outputs, ctx);
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) override {
        if (!this->_flag_init) {
            printf("init conv_act_pool first\n");
            return SaberNotInitialized;
        }
        SaberStatus state = _conv_act_func->dispatch(inputs, _vtensor_tmp);
        if (state != SaberSuccess) {
            return state;
        }
        return _pool_func->dispatch(_vtensor_tmp, outputs);
    }

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
