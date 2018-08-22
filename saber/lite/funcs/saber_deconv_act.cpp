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
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
    delete _conv_func;
}

SaberStatus SaberDeconvAct2D::load_param(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const ConvAct2DParam*)param;
    this->_flag_param = true;
    _conv_func->set_activation(_param->_flag_act);
    return _conv_func->load_param(&_param->_conv_param);
}

SaberStatus SaberDeconvAct2D::load_param(std::istream &stream, const float *weights) {
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
    int act_type;
    int flag_act;
    int w_offset;
    int b_offset;
    stream >> weights_size >> num_out >> group >> kw >> kh >> stride_w >> stride_h >> \
           pad_w >> pad_h >> dila_w >> dila_h >> flag_bias >> act_type >> flag_act >> w_offset >> b_offset;
    _param = new ConvAct2DParam(weights_size, num_out, group, kw, kh, \
        stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias>0, \
        (ActiveType)act_type, flag_act>0, \
        weights + w_offset, weights + b_offset);
    this->_flag_create_param = true;
    this->_flag_param = true;
    _conv_func->set_activation(flag_act);
    return _conv_func->load_param(&_param->_conv_param);
}
#if 0
SaberStatus SaberDeconvAct2D::load_param(FILE *fp, const float *weights) {
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
    int act_type;
    int flag_act;
    int w_offset;
    int b_offset;
    fscanf(fp, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
           &weights_size,
           &num_out,
           &group,
           &kw,
           &kh,
           &stride_w,
           &stride_h,
           &pad_w,
           &pad_h,
           &dila_w,
           &dila_h,
           &flag_bias,
           &act_type,
           &flag_act,
           &w_offset,
           &b_offset);
    _param = new ConvAct2DParam(weights_size, num_out, group, kw, kh, \
        stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias>0, \
        (ActiveType)act_type, flag_act>0, \
        weights + w_offset, weights + b_offset);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#endif
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
#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    _conv_func->set_op_name(this->get_op_name());
#endif
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