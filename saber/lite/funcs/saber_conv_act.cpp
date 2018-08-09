#include "saber/lite/funcs/saber_conv_act.h"
#include "saber/lite/net/saber_factory_lite.h"
namespace anakin{

namespace saber{

namespace lite{

SaberConvAct2D::SaberConvAct2D() {
    //_conv_func = new SaberConv2D;
    _conv_op = new SaberConv2D;
    _act_op = nullptr;
}

SaberConvAct2D::SaberConvAct2D(const ParamBase *param) {
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

SaberConvAct2D::~SaberConvAct2D() {
    // delete _conv_func;
    delete _conv_op;
    if(_act_op) {
        delete _act_op;
    }
}

SaberStatus SaberConvAct2D::load_param(const ParamBase *param) {
    _param = (const ConvAct2DParam*)param;
    this->_flag_param = true;
    _conv_op->set_activation(_param->_flag_act);
    return _conv_op->load_param(&_param->_conv_param);
}

SaberStatus SaberConvAct2D::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                 std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load conv_act param first\n");
        return SaberNotInitialized;
    }
    return _conv_op->compute_output_shape(inputs, outputs);
}

SaberStatus SaberConvAct2D::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *> &outputs,
                                 Context &ctx) {
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

SaberStatus SaberConvAct2D::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
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
REGISTER_LAYER_CLASS(SaberConvAct2D);
} //namespace lite

} //namespace saber

} //namespace anakin