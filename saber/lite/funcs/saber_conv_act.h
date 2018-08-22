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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H

#include "saber/lite/funcs/saber_conv.h"
#include "saber/lite/funcs/saber_activation.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberConvAct2D : public OpBase {
public:
    SaberConvAct2D() {
        //_conv_func = new SaberConv2D;
        _conv_op = new SaberConv2D;
        _act_op = nullptr;
    }

    SaberConvAct2D(const ParamBase* param) {
        _conv_op = new SaberConv2D;
        _param = (const ConvAct2DParam*)param;
        /*
        if (_param->_flag_act) {
            LCHECK_EQ(_param->_act_type, Active_relu, "active type must be relu");
        }
        */
        this->_flag_param = true;
        _conv_op->load_param(&_param->_conv_param);
    }

    virtual SaberStatus load_param(const ParamBase* param) override {
        _param = (const ConvAct2DParam*)param;
        this->_flag_param = true;
        _conv_op->set_activation(_param->_flag_act);
        return _conv_op->load_param(&_param->_conv_param);
    }

    ~SaberConvAct2D() {
       // delete _conv_func;
        delete _conv_op;
        if(_act_op) {
            delete _act_op;
        }
    }

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs)  override{
        if (!this->_flag_param) {
            printf("load conv_act param first\n");
            return SaberNotInitialized;
        }
        return _conv_op->compute_output_shape(inputs, outputs);
    }

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override{
        if (!this->_flag_param) {
            printf("load conv_act param first\n");
            return SaberNotInitialized;
        }
      //  _conv_func->set_activation(_param->_flag_act);
        this->_flag_init = true;

        _conv_op->init(inputs, outputs, ctx);
        if (_param->_flag_act) {
            if (_param->_act_type == Active_relu) {
                _conv_op->set_activation(_param->_flag_act);
             } else {
                if (_act_op == nullptr) {
                    _act_op = new SaberActivation;
                }
                _act_op->init(outputs, outputs, ctx);
            }
        }
        return SaberSuccess; //_conv_func->init(inputs, outputs, ctx);
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) override{
        if (!this->_flag_init) {
            printf("init conv_act first\n");
            return SaberNotInitialized;
        }
        SaberStatus state = _conv_op->dispatch(inputs, outputs);
        if (_act_op) {
            state = (SaberStatus)(state & _act_op->dispatch(outputs, outputs));
        }
        return state; //_conv_func->dispatch(inputs, outputs);
    }
private:
    const ConvAct2DParam* _param;
   // SaberConv2D* _conv_func;
    SaberConv2D* _conv_op{nullptr};
    SaberActivation* _act_op{nullptr};
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H
