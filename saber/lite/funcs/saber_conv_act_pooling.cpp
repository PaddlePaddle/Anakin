#include "saber/lite/funcs/saber_conv_act_pooling.h"
#include "saber/lite/net/saber_factory_lite.h"
namespace anakin{

namespace saber{

namespace lite{

SaberConvActPooling2D::SaberConvActPooling2D() {
    _pool_func = new SaberPooling;
    _conv_act_func = new SaberConvAct2D;
    _vtensor_tmp.push_back(&_tensor_tmp);
}

SaberConvActPooling2D::SaberConvActPooling2D(const ParamBase *param) {
    _pool_func = new SaberPooling;
    _conv_act_func = new SaberConvAct2D;
    _param = (const ConvActPool2DParam*)param;
    _conv_act_func->load_param(&_param->_conv_act_param);
    _pool_func->load_param(&_param->_pool_param);
    this->_flag_param = true;
}

SaberConvActPooling2D::~SaberConvActPooling2D() {
    delete _pool_func;
    delete _conv_act_func;
    _pool_func = nullptr;
    _conv_act_func = nullptr;
}

SaberStatus SaberConvActPooling2D::load_param(const ParamBase *param) {
    _param = (const ConvActPool2DParam*)param;
    this->_flag_param = true;
    SaberStatus state = _conv_act_func->load_param(&_param->_conv_act_param);
    if (state != SaberSuccess) {
        printf("load conv2d failed\n");
        return state;
    }
    return _pool_func->load_param(&_param->_pool_param);
}

SaberStatus SaberConvActPooling2D::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                        std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
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

SaberStatus SaberConvActPooling2D::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                        std::vector<Tensor<CPU, AK_FLOAT> *> &outputs,
                                        Context &ctx) {
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

SaberStatus SaberConvActPooling2D::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                            std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
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

REGISTER_LAYER_CLASS(SaberConvActPooling2D);
} //namespace lite

} //namespace saber

} //namespace anakin