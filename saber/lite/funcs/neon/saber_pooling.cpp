#include "saber/lite/funcs/saber_pooling.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/neon/impl/pooling_arm_impl.h"

namespace anakin {

namespace saber {

namespace lite{



SaberPooling::SaberPooling(PoolingType type, bool flag_global, int kernel_w, int kernel_h, \
    int stride_w, int stride_h, int pad_w, int pad_h) {

    _type = type;
    _is_global = flag_global;
    _kw = kernel_w;
    _kh = kernel_h;
    _stride_w = stride_w;
    _stride_h = stride_h;
    _pad_w = pad_w;
    _pad_h = pad_h;
}

SaberStatus SaberPooling::load_param(PoolingType type, bool flag_global, int kernel_w, int kernel_h, \
    int stride_w, int stride_h, int pad_w, int pad_h) {

    _type = type;
    _is_global = flag_global;
    _kw = kernel_w;
    _kh = kernel_h;
    _stride_w = stride_w;
    _stride_h = stride_h;
    _pad_w = pad_w;
    _pad_h = pad_h;
    return SaberSuccess;
}

SaberStatus SaberPooling::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    Shape output_shape = inputs[0]->valid_shape();

    int in_height = inputs[0]->height();
    int in_width = inputs[0]->width();

    int out_height;
    int out_width;
    if (_is_global) {
        out_height = 1;
        out_width = 1;
    } else {
        out_height = (in_height + 2 * _pad_h - _kh + _stride_h - 1) / _stride_h + 1;
        out_width = (in_width + 2 * _pad_w - _kw + _stride_w - 1) / _stride_w + 1;
    }

    output_shape.set_height(out_height);
    output_shape.set_width(out_width);

    return outputs[0]->set_shape(output_shape);
}

//template <>
SaberStatus SaberPooling::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    _ctx = ctx;

    if (_is_global) {
        _impl = pooling_global;
        return SaberSuccess;
    }

    if (_kw != _kh || _stride_w != _stride_h \
        || _stride_w != 2 || _pad_w != _pad_h || _pad_w > 1) {
        _impl = pooling_basic;
        return SaberSuccess;
    }

    if (_kw == 2) {
        if (_type == Pooling_max) {
            _impl = pooling2x2s2_max;
        } else {
            _impl = pooling2x2s2_ave;
        }
        return SaberSuccess;
    }

    if (_kw == 3) {
        if (_type == Pooling_max) {
            _impl = pooling3x3s2_max;
        } else {
            _impl = pooling3x3s2_ave;
        }
        return SaberSuccess;
    }

    _impl = pooling_basic;
    return SaberSuccess;
}

//template <>
SaberStatus SaberPooling::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                   std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    const float* din = inputs[0]->data();
    float* dout = outputs[0]->mutable_data();
    int num = inputs[0]->num();
    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();

    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();

    _impl(din, dout, num, chout, hout, wout, chin, hin, win, \
        _type, _is_global, _kw, _kh, \
        _stride_w, _stride_h, \
        _pad_w, _pad_h);
    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE