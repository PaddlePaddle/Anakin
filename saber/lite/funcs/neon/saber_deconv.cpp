#include "saber/lite/funcs/saber_deconv.h"
#ifdef USE_ARM_PLACE
#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#include "saber/lite/net/saber_factory_lite.h"
#include "saber/lite/funcs/neon/impl/sgemm_conv.h"
#include "saber/lite/funcs/neon/impl/sgemm_prepacked_int8.h"
#include "saber/lite/funcs/calibrate_lite.h"
#include "saber/lite/core/tensor_op_lite.h"
namespace anakin{

namespace saber{

namespace lite{

/**
 * \brief neon implementation to add bias and relu
 * @param tensor
 * @param bias
 * @param channel
 * @param channel_size
 */
template <typename Dtype>
void fill_bias_relu(Dtype* tensor, const Dtype* bias, int channel, int channel_size, bool flag_bias, bool flag_relu);

template <>
void fill_bias_relu(float* tensor, const float* bias, int channel, int channel_size, bool flag_bias, bool flag_relu) {
    float* data = tensor;
    if (flag_relu){
        for (int j = 0; j < channel; ++j) {
            float bias_data = flag_bias? bias[j] : 0.f;
            float32x4_t vbias = vdupq_n_f32(bias_data);
            float32x4_t vzero = vdupq_n_f32(0.f);
            int i = 0;
            for (; i < channel_size - 3; i += 4) {
                float32x4_t vdata = vld1q_f32(&data[i]);
                vdata = vaddq_f32(vdata, vbias);
                float32x4_t vmax = vmaxq_f32(vdata, vzero);
                vst1q_f32(data + i, vmax);
            }
            for (; i < channel_size; i++) {
                data[i] += bias_data;
                data[i] = data[i] > 0 ? data[i] : 0.f;
            }
            data += channel_size;
        }
    } else {
        for (int j = 0; j < channel; ++j) {
            float bias_data = flag_bias? bias[j] : 0.f;
            float32x4_t vbias = vdupq_n_f32(bias_data);
            int i = 0;
            for (; i < channel_size - 3; i += 4) {
                float32x4_t vdata = vld1q_f32(&data[i]);
                vdata = vaddq_f32(vdata, vbias);
                vst1q_f32(data + i, vdata);
            }
            for (; i < channel_size; i++) {
                data[i] += bias_data;
            }
            data += channel_size;
       }
    }
}

template <>
void fill_bias_relu(int* tensor, const int* bias, int channel, int channel_size, bool flag_bias, bool flag_relu) {
    int* data = tensor;
    if (flag_relu){
        for (int j = 0; j < channel; ++j) {
            int bias_data = flag_bias? bias[j] : 0;
            int32x4_t vbias = vdupq_n_s32(bias_data);
            int32x4_t vzero = vdupq_n_s32(0);
            int i = 0;
            for (; i < channel_size - 7; i += 8) {
                int32x4_t vdata1 = vld1q_s32(data + i);
                int32x4_t vdata2 = vld1q_s32(data + i + 4);
                vdata1 = vaddq_s32(vdata1, vbias);
                vdata2 = vaddq_s32(vdata2, vbias);
                int32x4_t vmax1 = vmaxq_s32(vdata1, vzero);
                int32x4_t vmax2 = vmaxq_s32(vdata2, vzero);
                vst1q_s32(data + i, vmax1);
                vst1q_s32(data + i + 4, vmax2);
            }
            for (; i < channel_size; i++) {
                data[i] += bias_data;
                data[i] = data[i] > 0 ? data[i] : 0;
            }
            data += channel_size;
        }
    } else {
        for (int j = 0; j < channel; ++j) {
            int bias_data = flag_bias? bias[j] : 0;
            int32x4_t vbias = vdupq_n_s32(bias_data);
            int i = 0;
            for (; i < channel_size - 7; i += 8) {
                int32x4_t vdata1 = vld1q_s32(data + i);
                int32x4_t vdata2 = vld1q_s32(data + i + 4);
                vdata1 = vaddq_s32(vdata1, vbias);
                vdata2 = vaddq_s32(vdata2, vbias);
                vst1q_s32(data + i, vdata1);
                vst1q_s32(data + i + 4, vdata2);
            }
            for (; i < channel_size; i++) {
                data[i] += bias_data;
            }
            data += channel_size;
        }
    }
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void col2im(const Dtype* data_col, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                Dtype* data_im) {
    memset(data_im, 0, height * width * channels * sizeof(Dtype));
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        data_col += output_w;
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                data_im[input_row * width + input_col] += *data_col;
                            }
                            data_col++;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

SaberDeconv2D::SaberDeconv2D() {
    _param = nullptr;
}

SaberDeconv2D::SaberDeconv2D(ParamBase* param) {
    _param = (Conv2DParam*)param;
    this->_flag_param = true;
}

SaberDeconv2D::~SaberDeconv2D() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberDeconv2D::load_param(ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (Conv2DParam*)(param);
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberDeconv2D::load_param(std::istream &stream, const float *weights) {
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
    bool flag_act;
    int act_type_i;
    float neg_slop;
    float act_coef;
    int act_channel_shared;
    int act_offset;

    stream >> weights_size >> num_out >> group >> kw >> kh >> stride_w >> stride_h >> \
           pad_w >> pad_h >> dila_w >> dila_h >> flag_bias >> w_offset >> b_offset >> \
            flag_eltwise >> flag_act >> act_type_i >> neg_slop >> act_coef >> act_channel_shared >> act_offset;
    ActiveType act_type = (ActiveType)act_type_i;
    _param = new Conv2DParam(weights_size, num_out, group, kw, kh, \
        stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias > 0, \
        weights + w_offset, weights + b_offset, flag_eltwise > 0, flag_act, \
        act_type, neg_slop, act_coef, act_channel_shared > 0, weights + act_offset);

    if (act_type != Active_relu) {
        LOGE("ERROR: active type %d is not supported now\n", (int)act_type);
        return SaberInvalidValue;
    }

    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberDeconv2D::set_op_precision(DataType ptype) {
    if (ptype == AK_FLOAT || ptype == AK_INT8) {
        _precision_type = ptype;
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}

SaberStatus SaberDeconv2D::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                                std::vector<Tensor<CPU> *> &outputs) {
    Shape output_shape = (inputs[0]->shape());

    if (!this->_flag_param) {
        LOGE("ERROR: load deconv param first\n");
        return SaberNotInitialized;
    }

    if (inputs[0]->dims() < 4) {
        LOGE("ERROR: using reshape2d to reshape a 1d conv?\n");
        return SaberInvalidValue;
    }

    output_shape.set_num(inputs[0]->num()); // N
    output_shape.set_channel(_param->_num_output); // K

    int kernel_extent_h = _param->_dila_h * (_param->_kh - 1) + 1;
    int output_dim_h = (inputs[0]->height() - 1) *
                       _param->_stride_h + kernel_extent_h - 2 * _param->_pad_h;
    int kernel_extent_w = _param->_dila_w * (_param->_kw - 1) + 1;
    int output_dim_w = (inputs[0]->width() - 1) *
                       _param->_stride_w + kernel_extent_w - 2 * _param->_pad_w;

    output_shape.set_height(output_dim_h);
    output_shape.set_width(output_dim_w);

    return outputs[0]->set_shape(output_shape);
}

SaberStatus SaberDeconv2D::init(const std::vector<Tensor<CPU> *> &inputs, \
    std::vector<Tensor<CPU> *> &outputs, Context &ctx) {

    if (!this->_flag_param) {
        LOGE("ERROR: load deconv param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();

    if (chin != chout || _param->_group != chin) {
        LCHECK_EQ(chin % _param->_group, 0, "ERROR: input channel or group size error\n");
        LCHECK_EQ(chout % _param->_group, 0, "ERROR: output channel or group size error\n");
    }
    //! deconv weights layout: chin * chout * kh * kw
    _m = chout * _param->_kw * _param->_kh / _param->_group;
    _n = hin * win;
    _k = chin / _param->_group;

    _ctx->workspace_extend(Shape(_param->_group * _m * _n));

    if (this->get_op_precision() == AK_FLOAT) {
        int hblock = get_hblock(this->_ctx->get_arch());
        int m_roundup = hblock * ((_m + hblock - 1) / hblock);
        int group_size_round_up = ((m_roundup * _k + 15) / 16) * 16;
        Tensor<CPU> tmp_weights;
        tmp_weights.reshape(Shape(group_size_round_up * _param->_group));
        for (int g = 0; g < _param->_group; ++g) {
            const float* weights_group = static_cast<const float*>(_param->_weights.data()) + g * _m * _k;
            float* weights_trans_ptr = static_cast<float*>(tmp_weights.mutable_data()) + g * group_size_round_up;
            prepackA(weights_trans_ptr, weights_group, _m, 0, _m, 0, _k, true, this->_ctx);
        }
        _param->_weights.reshape(Shape(group_size_round_up * _param->_group));
        _param->_weights.copy_from(tmp_weights);
#if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
        printf("Deconv: USE GEMM, numout=%d, chin=%d, kernel=%d, stride=%d, pad=%d, group=%d, win=%d, hin=%d\n", \
            chout, chin, _param->_kw, _param->_stride_w, _param->_pad_w, _param->_group, win, hin);
#endif
        this->_flag_init = true;
        return SaberSuccess;
    } else if (this->get_op_precision() == AK_INT8) {
        _tmp_out.set_dtype(AK_INT32);
        _tmp_out.reshape(outputs[0]->valid_shape());
        Shape act_shape = _param->_weights.valid_shape();
        int tmpc = act_shape[1];
        act_shape[1] = act_shape[0];
        act_shape[0] = tmpc;
        _param->_weights.set_shape(act_shape);
        trans_fp32_weights_to_int8_inplace_gemm(_param->_weights, 63.f, true, _param->_group, _ctx);
        _w_scale = _param->_weights.get_scale();
        trans_fp32_bias_to_int32_inplace(_param->_bias, inputs[0]->get_scale()[0], _w_scale, _ctx);
        this->_flag_init = true;
        return SaberSuccess;
    } else {
        LOGE("ERROR: deconv unsupported precision type: %d\n", (int)this->get_op_precision());
        return SaberUnImplError;
    }
    return SaberSuccess;
}


SaberStatus SaberDeconv2D::dispatch(const std::vector<Tensor<CPU> *> &inputs, \
    std::vector<Tensor<CPU> *> &outputs) {

    if (!this->_flag_init) {
        LOGE("ERROR: init deconv first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();

    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();

    int group = _param->_group;

    int group_size_in = win * hin * chin / group;
    int group_size_out = wout * hout * chout / group;
    int group_size_coldata = _m * _n;

    int hblock = get_hblock(this->_ctx->get_arch());
    int m_roundup = hblock * ((_m + hblock - 1) / hblock);
    int group_size_weights = ((m_roundup * _k + 15) / 16) * 16;

    bool flag_1x1s1p1 = (_param->_kw == 1) && (_param->_kh == 1) && (_param->_stride_h == 1) && \
        (_param->_stride_w == 1) && (_param->_pad_w == 0) && (_param->_pad_h == 0) && \
        (_param->_dila_w == 1) && (_param->_dila_h == 1);

    const void* din = nullptr;
    void* dout = nullptr;

    //! prepare input data
    if (this->get_op_precision() == AK_INT8) {
        if (inputs[0]->get_dtype() != AK_INT8) {
//            LOGE("conv int8 trans input, fp32 to int8\n");
            _tmp_in.set_dtype(AK_INT8);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_fp32_to_int8(*inputs[0], _tmp_in, _ctx);
            din = _tmp_in.data();
        } else {
//            LOGE("conv int8 trans input, no trans\n");
            din = inputs[0]->data();
        }
    } else if (this->get_op_precision() == AK_FLOAT) {
        if (inputs[0]->get_dtype() != AK_FLOAT) {
//            LOGE("conv fp32 trans input, int8 to fp32\n");
            _tmp_in.set_dtype(AK_FLOAT);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_int8_to_fp32(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], _ctx);
            din = _tmp_in.data();
        } else {
//            LOGE("conv fp32 trans input, no trans\n");
            din = inputs[0]->data();
        }
    } else {
        LOGE("ERROR: unsupported input data type!!\n");
        return SaberInvalidValue;
    }

    dout = outputs[0]->mutable_data();

    if (this->get_op_precision() == AK_INT8) {
        dout = _tmp_out.mutable_data();
    } else if (this->get_op_precision() == AK_FLOAT) {
        dout = outputs[0]->mutable_data();
    } else {
        LOGE("ERROR: unsupported precision type!!\n");
        return SaberInvalidValue;
    }

    if (this->get_op_precision() == AK_FLOAT) {
        //! do nothing
        if (outputs[0]->get_dtype() != AK_FLOAT) {
            LOGE("ERROR: unsupported precision type!!\n");
            return SaberInvalidValue;
        }
    }
    const void* weights = _param->_weights.data();;
    if (this->get_op_precision() == AK_FLOAT) {
        for (int i = 0; i < num; ++i) {
            const float* din_batch = static_cast<const float*>(din) + i * chin * hin * win;
            float* dout_batch = static_cast<float*>(dout) + i * chout * hout * wout;

            float* col_data = static_cast<float*>(_ctx->get_work_space()) + _ctx->l2_cache_size() / sizeof(float);
            if (flag_1x1s1p1) {
                col_data = dout_batch;
            }
            for (int g = 0; g < group; ++g) {
                const float* din_group = din_batch + g * group_size_in;
                const float* weights_group = static_cast<const float*>(weights) + g * group_size_weights;
                float* coldata_group = col_data + g * group_size_coldata;
                sgemm_prepack(weights_group, din_group, nullptr, coldata_group, _m, _n, _k, \
                    false, (_param->_flag_act && !_param->_bias_term), false, this->_ctx);
            }

            if (!flag_1x1s1p1) {
                col2im(col_data, chout, hout, wout, _param->_kh, _param->_kw, _param->_pad_h, _param->_pad_w, \
                _param->_stride_h, _param->_stride_w, _param->_dila_h, _param->_dila_w, dout_batch);
            }

            //! add bias
            if (_param->_bias_term) {
                fill_bias_relu<float>(dout_batch, static_cast<const float*>(_param->_bias.data()), \
                chout, wout * hout, _param->_bias_term, _param->_flag_act);
            }
        }
    } else if (this->get_op_precision() == AK_INT8) {
        for (int i = 0; i < num; ++i) {
            const char* din_batch = static_cast<const char*>(din) + i * chin * hin * win;
            int* dout_batch = static_cast<int*>(dout) + i * chout * hout * wout;

            int* col_data = static_cast<int*>(_ctx->get_work_space()) + _ctx->l2_cache_size() / sizeof(int);
            if (flag_1x1s1p1) {
                col_data = dout_batch;
            }
            for (int g = 0; g < group; ++g) {
                const char* din_group = din_batch + g * group_size_in;
                const char* weights_group = static_cast<const char*>(weights) + g * group_size_weights;
                int* coldata_group = col_data + g * group_size_coldata;
                sgemm_prepack_int8(weights_group, din_group, nullptr, coldata_group, _m, _n, _k, \
                    false, (_param->_flag_act && !_param->_bias_term), false, this->_ctx);
            }

            if (!flag_1x1s1p1) {
                col2im(col_data, chout, hout, wout, _param->_kh, _param->_kw, _param->_pad_h, _param->_pad_w, \
                _param->_stride_h, _param->_stride_w, _param->_dila_h, _param->_dila_w, dout_batch);
            }

            //! add bias
            if (_param->_bias_term) {
                fill_bias_relu<int>(dout_batch, static_cast<const int*>(_param->_bias.data()), \
                chout, wout * hout, _param->_bias_term, _param->_flag_act);
            }
        }
    } else {
        LOGE("ERROR: deconv unsupported precision type: %d\n", (int)this->get_op_precision());
        return SaberUnImplError;
    }

    //! trans output data
    if (this->get_op_precision() == AK_INT8) {
        if (outputs[0]->get_dtype() == AK_INT8) {
//            LOGE("conv trans output, int32 to int8\n");
            trans_tensor_int32_to_int8(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], _w_scale, _ctx);
        } else if (outputs[0]->get_dtype() == AK_FLOAT) {
//            LOGE("conv trans output, int32 to fp32\n");
            trans_tensor_int32_to_fp32(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], _w_scale, _ctx);
        } else {
            LOGE("ERROR: unsupported precision type!!\n");
            return SaberInvalidValue;
        }
    }

#ifdef ENABLE_OP_TIMER
    float op_macs = _param->_kw * _param->_kh * \
        num * chout * hout * wout * chin / _param->_group;
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    GOPS ops;
    ops.ts = ts;
    ops.ops = op_macs;
    printf("deconv %s: time: %f ms, %f GOPs, %f GOPS\n", this->_op_name.c_str(), ts, 1e-9f * op_macs, 0.000001 * op_macs / ts);
    OpTimer::add_timer("deconvolution", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberDeconv2D);
} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


