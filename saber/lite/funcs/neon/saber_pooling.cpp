#include "saber/lite/funcs/saber_pooling.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/neon/impl/pooling_arm_impl.h"

namespace anakin {

namespace saber {

namespace lite{

SaberPooling::SaberPooling(const ParamBase *param) {
    _param = (const PoolParam*)param;
    this->_flag_param = true;
}

SaberPooling::~SaberPooling() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberPooling::load_param(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const PoolParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberPooling::load_param(FILE *fp, const float *weights) {
    int type;
    int g_pool;
    int kw;
    int kh;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    fscanf(fp, "%d,%d,%d,%d,%d,%d,%d,%d\n", &type, &g_pool, &kw, &kh, &stride_w, &stride_h, &pad_w, &pad_h);
    PoolingType ptype = (PoolingType)type;
    _param = new PoolParam(ptype, g_pool>0, kw, kh, stride_w, stride_h, pad_w, pad_h);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberPooling::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_param) {
        printf("load pooling param first\n");
        return SaberNotInitialized;
    }

    Shape output_shape = inputs[0]->valid_shape();

    int in_height = inputs[0]->height();
    int in_width = inputs[0]->width();

    int out_height;
    int out_width;
    if (_param->_flag_global) {
        out_height = 1;
        out_width = 1;
    } else {
        out_height = (in_height + 2 * _param->_pool_pad_h - _param->_pool_kh + _param->_pool_stride_h - 1) / _param->_pool_stride_h + 1;
        out_width = (in_width + 2 * _param->_pool_pad_w - _param->_pool_kw + _param->_pool_stride_w - 1) / _param->_pool_stride_w + 1;
    }

    output_shape.set_height(out_height);
    output_shape.set_width(out_width);

    return outputs[0]->set_shape(output_shape);
}

//template <>
SaberStatus SaberPooling::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        printf("load pooling param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;

    if (_param->_flag_global) {
        _impl = pooling_global;
        this->_flag_init = true;
        return SaberSuccess;
    }

    if (_param->_pool_kw == 2 && _param->_pool_kh == 2 && \
            _param->_pool_stride_w == 2 && _param->_pool_pad_h == 2 && \
            _param->_pool_pad_h == 0 && _param->_pool_pad_w == 0) {
        if (_param->_pool_type == Pooling_max) {
            _impl = pooling2x2s2_max;
        } else {
            _impl = pooling2x2s2_ave;
        }
        this->_flag_init = true;
        return SaberSuccess;
    }

    if (_param->_pool_kw == 3 && _param->_pool_kh == 3) {
        if (_param->_pool_stride_h == 1 && _param->_pool_stride_w == 1 && \
            _param->_pool_pad_h == 1 && _param->_pool_pad_w == 1) {
            if (_param->_pool_type == Pooling_max) {
                _impl = pooling3x3s1p1_max;
            } else {
                _impl = pooling3x3s1p1_ave;
            }
            this->_flag_init = true;
            return SaberSuccess;
        }

        if (_param->_pool_stride_w == 2 && _param->_pool_stride_h == 2) {
            if (_param->_pool_pad_w == 0 &&  _param->_pool_pad_h == 0) {
                if (_param->_pool_type == Pooling_max) {
                    _impl = pooling3x3s2p0_max;
                } else {
                    _impl = pooling3x3s2p0_ave;
                }
                this->_flag_init = true;
                return SaberSuccess;
            }
            if (_param->_pool_pad_w == 1 && _param->_pool_pad_h == 1) {
                if (_param->_pool_type == Pooling_max) {
                    _impl = pooling3x3s2p1_max;
                } else {
                    _impl = pooling3x3s2p1_ave;
                }
                this->_flag_init = true;
                return SaberSuccess;
            }
        }
    }

    _impl = pooling_basic;

    this->_flag_init = true;

    return SaberSuccess;
}

//template <>
SaberStatus SaberPooling::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                   std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_init) {
        printf("init pool first\n");
        return SaberNotInitialized;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

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
        _param->_pool_type, _param->_flag_global, _param->_pool_kw, _param->_pool_kh, \
        _param->_pool_stride_w, _param->_pool_stride_h, \
        _param->_pool_pad_w, _param->_pool_pad_h);

#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("pooling time: %f\n", ts);
    OpTimer::add_timer("pooling", ts);
    OpTimer::add_timer("total", ts);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberPooling);
} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE