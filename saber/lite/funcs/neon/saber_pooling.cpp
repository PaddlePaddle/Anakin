#include "saber/lite/funcs/saber_pooling.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/neon/impl/pooling_arm_impl.h"

namespace anakin {

namespace saber {

namespace lite{

SaberPooling::SaberPooling(ParamBase *param) {
    _param = (PoolParam*)param;
    this->_flag_param = true;
}

SaberPooling::~SaberPooling() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberPooling::load_param(ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (PoolParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberPooling::load_param(std::istream &stream, const float *weights) {
    int type;
    int g_pool;
    int kw;
    int kh;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    stream >> type >> g_pool >> kw >> kh >> stride_w >> stride_h >> pad_w >> pad_h;
    PoolingType ptype = (PoolingType)type;
    _param = new PoolParam(ptype, g_pool > 0, kw, kh, stride_w, stride_h, pad_w, pad_h);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberPooling::set_op_precision(DataType ptype) {
    if (ptype == AK_FLOAT || ptype == AK_INT8) {
        _precision_type = ptype;
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}

SaberStatus SaberPooling::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                               std::vector<Tensor<CPU> *> &outputs) {

    if (!this->_flag_param) {
        LOGE("ERROR: load pooling param first\n");
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
SaberStatus SaberPooling::init(const std::vector<Tensor<CPU> *> &inputs,
                               std::vector<Tensor<CPU> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        LOGE("ERROR: load pooling param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;

    DataType op_type = this->get_op_precision();
    switch (op_type){
        case AK_FLOAT:
            _tmp_out.set_dtype(AK_FLOAT);
            _tmp_out.reshape(outputs[0]->valid_shape());
            if (_param->_flag_global) {
                _impl = pooling_global;
                this->_flag_init = true;
                return SaberSuccess;
            }

            if (_param->_pool_kw == 2 && _param->_pool_kh == 2 && \
                    _param->_pool_stride_w == 2 && _param->_pool_stride_h == 2 && \
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
            break;

        case AK_INT8:
            _tmp_out.set_dtype(AK_INT8);
            _tmp_out.reshape(outputs[0]->valid_shape());
            if (_param->_flag_global) {
                _impl = pooling_global_int8;
                this->_flag_init = true;
                return SaberSuccess;
            }

            if (_param->_pool_kw == 2 && _param->_pool_kh == 2 && \
                    _param->_pool_stride_w == 2 && _param->_pool_stride_h == 2 && \
                    _param->_pool_pad_h == 0 && _param->_pool_pad_w == 0) {
                if (_param->_pool_type == Pooling_max) {
                    _impl = pooling2x2s2_max_int8;
                } else {
                    _impl = pooling2x2s2_ave_int8;
                }
                this->_flag_init = true;
                return SaberSuccess;
            }

            if (_param->_pool_kw == 3 && _param->_pool_kh == 3) {
                if (_param->_pool_stride_h == 1 && _param->_pool_stride_w == 1 && \
                    _param->_pool_pad_h == 1 && _param->_pool_pad_w == 1) {
                    if (_param->_pool_type == Pooling_max) {
                        _impl = pooling3x3s1p1_max_int8;
                    } else {
                        _impl = pooling_basic_int8;
                    }
                    this->_flag_init = true;
                    return SaberSuccess;
                }

                if (_param->_pool_stride_w == 2 && _param->_pool_stride_h == 2) {
                    if (_param->_pool_pad_w == 0 &&  _param->_pool_pad_h == 0) {
                        if (_param->_pool_type == Pooling_max) {
                            _impl = pooling3x3s2p0_max_int8;
                        } else {
                            _impl = pooling_basic_int8;
                        }
                        this->_flag_init = true;
                        return SaberSuccess;
                    }
                    if (_param->_pool_pad_w == 1 && _param->_pool_pad_h == 1) {
                        if (_param->_pool_type == Pooling_max) {
                            _impl = pooling3x3s2p1_max_int8;
                        } else {
                            _impl = pooling_basic_int8;
                        }
                        this->_flag_init = true;
                        return SaberSuccess;
                    }
                }
            }
            _impl = pooling_basic_int8;
            break;

        default:
            LOGF("data type: %d is unsupported now", (int)op_type);
    }

    this->_flag_init = true;

    return SaberSuccess;
}

//template <>
SaberStatus SaberPooling::dispatch(const std::vector<Tensor<CPU> *> &inputs,
                                   std::vector<Tensor<CPU> *> &outputs) {

    if (!this->_flag_init) {
        LOGE("ERROR: init pool first\n");
        return SaberNotInitialized;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    //const void* din = static_cast<const void*>(inputs[0]->data());
    //void* dout = static_cast<void*>(outputs[0]->mutable_data());
    const void* din = nullptr;
    void* dout = nullptr;
    int num = inputs[0]->num();
    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();

    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    DataType tensor_in_type = inputs[0]->get_dtype();
    DataType op_type = this->get_op_precision();
    if (op_type == AK_INT8){
        if (tensor_in_type == AK_FLOAT){
            _tmp_in.set_dtype(AK_INT8);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_fp32_to_int8(*inputs[0], _tmp_in, _ctx);
            din = _tmp_in.data();
        } else {
            din = inputs[0]->data();
        }
    } else if (op_type == AK_FLOAT){
        if (tensor_in_type == AK_INT8){
            _tmp_in.set_dtype(AK_FLOAT);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_int8_to_fp32(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], _ctx);
            din = _tmp_in.data();
        } else {
            din = inputs[0]->data();
        }
    } else {
        LOGE("ERROR: unsupported precision type!!\n");
        return SaberInvalidValue;
    }

    DataType tensor_out_type = outputs[0]->get_dtype();
    if (op_type == AK_INT8 && tensor_out_type == AK_INT8) {
        dout = outputs[0]->mutable_data();
    } else if (op_type == AK_INT8 && tensor_out_type == AK_FLOAT){
        dout = _tmp_out.mutable_data();
    } else if (op_type == AK_FLOAT) {
        dout = outputs[0]->mutable_data();
    } else {
        LOGE("ERROR: unsupported precision type!!\n");
        return SaberInvalidValue;
    }

    if (op_type == AK_FLOAT) {
        //! do nothing
        if (outputs[0]->get_dtype() != AK_FLOAT) {
            LOGE("ERROR: unsupported precision type!!\n");
            return SaberInvalidValue;
        }
    }
    _impl(din, dout, num, chout, hout, wout, chin, hin, win, \
        _param->_pool_type, _param->_flag_global, _param->_pool_kw, _param->_pool_kh, \
        _param->_pool_stride_w, _param->_pool_stride_h, \
        _param->_pool_pad_w, _param->_pool_pad_h);

    if (op_type == AK_INT8) {
        if (tensor_out_type == AK_FLOAT) {
            trans_tensor_int8_to_fp32(_tmp_out, *outputs[0], outputs[0]->get_scale()[0], _ctx);
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    GOPS ops;
    // fixme
    ops.ops = 0;
    ops.ts = ts;
    LOGI("pooling %s: time: %f\n", this->_op_name.c_str(), ts);
    OpTimer::add_timer("pooling", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberPooling);
} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE
