#include "saber/lite/funcs/saber_deconv_act.h"
#include "saber/lite/net/saber_factory_lite.h"
namespace anakin{

namespace saber{

namespace lite{
SaberDeconvAct2D::SaberDeconvAct2D() {
    _conv_func = new SaberDeconv2D;
}

SaberDeconvAct2D::SaberDeconvAct2D(const ParamBase *param) {
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

SaberDeconvAct2D::~SaberDeconvAct2D() {
    delete _conv_func;
}

SaberStatus SaberDeconvAct2D::load_param(const ParamBase *param) {
    _param = (const ConvAct2DParam*)param;
    this->_flag_param = true;
    _conv_func->set_activation(_param->_flag_act);
    return _conv_func->load_param(&_param->_conv_param);
}

SaberStatus SaberDeconvAct2D::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                   std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load conv_act param first\n");
        return SaberNotInitialized;
    }
    return _conv_func->compute_output_shape(inputs, outputs);
}

SaberStatus SaberDeconvAct2D::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                   std::vector<Tensor<CPU, AK_FLOAT> *> &outputs,
                                   Context &ctx) {
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

SaberStatus SaberDeconvAct2D::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                       std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_init) {
        printf("init conv_act first\n");
        return SaberNotInitialized;
    }
    return _conv_func->dispatch(inputs, outputs);
}

REGISTER_LAYER_CLASS(SaberDeconvAct2D);
} //namespace lite

} //namespace saber

} //namespace anakin