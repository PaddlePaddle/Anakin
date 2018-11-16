#include "saber/lite/funcs/saber_conv_pooling.h"
#include "saber/lite/net/saber_factory_lite.h"
#include "saber/lite/core/tensor_op_lite.h"
namespace anakin{

namespace saber{

namespace lite{

SaberConvPooling2D::SaberConvPooling2D() {
    _pool_func = new SaberPooling;
    _conv_func = new SaberConv2D;
    _vtensor_tmp.push_back(&_tensor_tmp);
}

SaberConvPooling2D::SaberConvPooling2D(ParamBase *param) {
    _pool_func = new SaberPooling;
    _conv_func = new SaberConv2D;
    _param = (ConvPool2DParam*)param;
    _conv_func->load_param(&_param->_conv_param);
    _pool_func->load_param(&_param->_pool_param);
    this->_flag_param = true;
}

SaberConvPooling2D::~SaberConvPooling2D() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
    delete _pool_func;
    delete _conv_func;
    _pool_func = nullptr;
    _conv_func = nullptr;
}

SaberStatus SaberConvPooling2D::load_param(ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (ConvPool2DParam*)param;
    this->_flag_param = true;
    SaberStatus state = _conv_func->load_param(&_param->_conv_param);
    if (state != SaberSuccess) {
        LOGE("load conv2d failed\n");
        return state;
    }
    return _pool_func->load_param(&_param->_pool_param);
}

SaberStatus SaberConvPooling2D::load_param(std::istream &stream, const float *weights) {
    int weights_size;
    int num_out;
    int group;
    int kw;
    int kh;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int dila_w;
    int dila_h;
    int flag_bias;
    int w_offset;
    int b_offset;
    int flag_eltwise;
    int flag_act;
    int act_type;
    float act_neg_slope;
    float act_coef;
    int act_channel_shared;
    int act_w_offset;
    int ptype;
    int g_pool;
    int pkw;
    int pkh;
    int pstride_w;
    int pstride_h;
    int ppad_w;
    int ppad_h;

    stream >> weights_size >> num_out >> group >> kw >> kh >> stride_w >> stride_h >> pad_w >> pad_h >> \
           dila_w >> dila_h >> flag_bias >> w_offset >> b_offset >> flag_eltwise >> flag_act >> act_type >> \
           act_neg_slope >> act_coef >> act_channel_shared >> act_w_offset >> \
           ptype >> g_pool >> pkw >> pkh >> pstride_w >> pstride_h >> ppad_w >> ppad_h ;
    ActiveType atype = static_cast<ActiveType>(act_type);
    PoolingType pool_type = static_cast<PoolingType>(ptype);
    _param = new ConvPool2DParam(weights_size, num_out, group, kw, kh, stride_w, stride_h, \
        pad_w, pad_h, dila_w, dila_h, flag_bias>0, weights + w_offset, weights + b_offset, \
        flag_eltwise > 0, flag_act > 0, atype, act_neg_slope, act_coef, act_channel_shared > 0, weights + act_w_offset, \
        pool_type, g_pool>0, pkw, pkh, pstride_w, pstride_h, ppad_w, ppad_h);
    this->_flag_create_param = true;
    this->_flag_param = true;
    SaberStatus state = _conv_func->load_param(&_param->_conv_param);
    if (state != SaberSuccess) {
        LOGE("load conv2d failed\n");
        return state;
    }
    return _pool_func->load_param(&_param->_pool_param);
}

SaberStatus SaberConvPooling2D::set_op_precision(DataType ptype) {
    auto flag = _conv_func->set_op_precision(ptype);
    flag = flag & _pool_func->set_op_precision(ptype);
    return flag;
}

SaberStatus SaberConvPooling2D::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                                        std::vector<Tensor<CPU> *> &outputs) {
    if (!this->_flag_param) {
        LOGE("load conv_act_pool param first\n");
        return SaberNotInitialized;
    }

    SaberStatus state = _conv_func->compute_output_shape(inputs, _vtensor_tmp);
    if (state != SaberSuccess) {
        return state;
    }
    _tensor_tmp.reshape(_tensor_tmp.valid_shape());
    return _pool_func->compute_output_shape(_vtensor_tmp, outputs);
}

SaberStatus SaberConvPooling2D::init(const std::vector<Tensor<CPU> *> &inputs,
                                        std::vector<Tensor<CPU> *> &outputs,
                                        Context &ctx) {
    if (!this->_flag_param) {
        LOGE("load conv_act_pool param first\n");
        return SaberNotInitialized;
    }
#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    _conv_func->set_op_name(this->get_op_name());
#endif
    _tensor_tmp.reshape(_tensor_tmp.valid_shape());
    this->_flag_init = true;
    SaberStatus state = _conv_func->init(inputs, _vtensor_tmp, ctx);
    if (state != SaberSuccess) {
        return state;
    }
    return _pool_func->init(_vtensor_tmp, outputs, ctx);
}

SaberStatus SaberConvPooling2D::dispatch(const std::vector<Tensor<CPU> *> &inputs,
                                            std::vector<Tensor<CPU> *> &outputs) {
    if (!this->_flag_init) {
        LOGE("init conv_act_pool first\n");
        return SaberNotInitialized;
    }
    SaberStatus state = _conv_func->dispatch(inputs, _vtensor_tmp);
    if (state != SaberSuccess) {
        return state;
    }
    return _pool_func->dispatch(_vtensor_tmp, outputs);
}

REGISTER_LAYER_CLASS(SaberConvPooling2D);
} //namespace lite

} //namespace saber

} //namespace anakin