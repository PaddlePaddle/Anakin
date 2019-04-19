#include "saber/funcs/impl/arm/saber_deconv.h"
#include "saber/funcs/impl/arm/neon/impl/sgemm_prepacked.h"
#include "saber/funcs/impl/arm/neon/impl/gemm_prepacked_int8.h"
#include "saber/funcs/type_trans.h"
namespace anakin{
namespace saber {


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

/******************************************  Deconv Precision Is Float ******************************************/

template<>
SaberStatus SaberDeconv2D<ARM, AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx) {

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
    _kw = param.weight()->width();
    _kh = param.weight()->height();

    if (chin != chout || param.group != chin) {
        CHECK_EQ(chin % param.group, 0) <<  "ERROR: input channel or group size error";
        CHECK_EQ(chout % param.group, 0) <<  "ERROR: output channel or group size error";
    }
    //! deconv weights layout: chin * chout * kh * kw
    _m = chout * _kw * _kh / param.group;
    _n = hin * win;
    _k = chin / param.group;

    _ctx->workspace_extend(Shape({1, 1, 1, param.group * _m * _n}));

    Tensor<ARM> tmp_w;
    prepackA(tmp_w, *param.weight(), _m, _k, param.group, true, this->_ctx);
    param.weight()->reshape(tmp_w.valid_shape());
    param.weight()->copy_from(tmp_w);
    return SaberSuccess;
}

template<>
SaberStatus SaberDeconv2D<ARM, AK_FLOAT>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDeconv2D<ARM, AK_FLOAT>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ConvParam<ARM> &param) {
    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();

    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    int group = param.group;

#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    int kw = param.weight()->width();
    int kh = param.weight()->height();
    int pw = param.pad_w;
    int sw = param.stride_w;
    int dw = param.dilation_w;
    int ph = param.pad_h;
    int sh = param.stride_h;
    int dh = param.dilation_h;
    LOG(INFO) << "conv param: " << " img_num = " << num << " in_channels = " << chin \
        << " img_h = " << hin << " img_w = " << win << " group = " << group \
        << " pad_width = " << pw << " pad_height = " << ph << " stride_width = " \
        << sw << " stride_height = " << sh << " dilation_w = " << dw \
        << " dilation_h = " << dh << " kernel_w = " << kw << " kernel_h = " \
        << kh << " out_channels = " << chout;
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif

    bool flag_relu = false;
    bool flag_bias = param.bias()->size() > 0;
    if (param.activation_param.has_active){
        if (param.activation_param.active == Active_relu){
            flag_relu = true;
        }
    }

    int group_size_in = win * hin * chin / group;
    int group_size_out = wout * hout * chout / group;
    int group_size_coldata = _m * _n;
    int hblock = get_hblock(this->_ctx->get_arch());
    int m_roundup = hblock * ((_m + hblock - 1) / hblock);
    int group_size_weights = ((m_roundup * _k + 15) / 16) * 16;

    bool flag_1x1s1p1 = (_kw == 1) && (_kh == 1) && (param.stride_h == 1) && \
        (param.stride_w == 1) && (param.pad_w == 0) && (param.pad_h == 0) && \
        (param.dilation_w == 1) && (param.dilation_h == 1);

    const float* din = inputs[0]->data();
    float* dout = outputs[0]->mutable_data();
    const float* weights = static_cast<const float*>(param.weight()->data());

    for (int i = 0; i < num; ++i) {
        const float* din_batch = din + i * chin * hin * win;
        float* dout_batch = dout + i * chout * hout * wout;

        float* col_data = static_cast<float*>(_ctx->get_work_space()) + _ctx->get_l2_cache_size() / sizeof(float);
        if (flag_1x1s1p1) {
            col_data = dout_batch;
        }
        for (int g = 0; g < group; ++g) {
            const float* din_group = din_batch + g * group_size_in;
            const float* weights_group = weights + g * group_size_weights;
            float* coldata_group = col_data + g * group_size_coldata;
            sgemm_prepack(weights_group, din_group, nullptr, coldata_group, _m, _n, _k, \
                false, /*false*/flag_relu && (!flag_bias), false, this->_ctx);
        }
        if (!flag_1x1s1p1) {
            col2im(col_data, chout, hout, wout, _kh, _kw, param.pad_h, param.pad_w, \
            param.stride_h, param.stride_w, param.dilation_h, param.dilation_w, dout_batch);
        }
        //! add bias
        if (flag_bias){
            fill_bias_relu<float>(dout_batch, static_cast<const float*>(param.bias()->data()), \
                chout, wout * hout, flag_bias, flag_relu);
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    float op_macs = 2.f * kw * kh * num * chout * wout * hout * chin / group;
    LOG(INFO) << "Deconvlution fp32: " << this->_op_name.c_str() << ", time: " << ts << \
        ", GOPs: " << 1e-9f * op_macs << ", GOPS: " << 0.000001 * op_macs / ts;
    GOPS ops;
    //fixme
    ops.ops = op_macs;
    ops.ts = ts;
    OpTimer::add_timer("Deconvlution", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}

/******************************************  Deconv Precision Is Int8 ******************************************/

template<>
SaberStatus SaberDeconv2D<ARM, AK_INT8>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx) {

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
    _kw = param.weight()->width();
    _kh = param.weight()->height();

    if (chin != chout || param.group != chin) {
        CHECK_EQ(chin % param.group, 0) <<  "ERROR: input channel or group size error";
        CHECK_EQ(chout % param.group, 0) <<  "ERROR: output channel or group size error";
    }
    //! deconv weights layout: chin * chout * kh * kw
    _m = chout * _kw * _kh / param.group;
    _n = hin * win;
    _k = chin / param.group;

    _ctx->workspace_extend(Shape({1, 1, 1, param.group * _m * _n}));

    Tensor<ARM> tmp_w;
    prepackA_int8(tmp_w, *param.weight(), _m, _k, param.group, true, this->_ctx);
    param.weight()->reshape(tmp_w.valid_shape());
    param.weight()->copy_from(tmp_w);
    //! init int32 output tmp
    if (outputs[0]->get_dtype() != AK_INT32) {
        _tmp_out.set_dtype(AK_INT32);
        _tmp_out.reshape(outputs[0]->valid_shape());
    }
    return SaberSuccess;
}

template<>
SaberStatus SaberDeconv2D<ARM, AK_INT8>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
        this->_ctx = &ctx;
        //! trans weights to int8
        if (trans_weights_dtype(*param.mutable_weight(), AK_INT8, 127.f, DECONV_TYPE, param.group) != SaberSuccess) {
            LOG(ERROR) << "ERROR: deconv trans weights to int8 failed";
            return SaberInvalidValue;
        }

        //! trans bias to int32
        if (param.bias()->size() > 0) {
            trans_fp32_bias_to_int32(*param.mutable_bias(), *param.mutable_bias(), inputs[0]->get_scale()[0], \
                param.weight()->get_scale());
        }
        return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDeconv2D<ARM, AK_INT8>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ConvParam<ARM> &param) {
    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();

    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    int group = param.group;
#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    int kw = param.weight()->width();
    int kh = param.weight()->height();
    int pw = param.pad_w;
    int sw = param.stride_w;
    int dw = param.dilation_w;
    int ph = param.pad_h;
    int sh = param.stride_h;
    int dh = param.dilation_h;
    LOG(INFO) << "conv param: " << " img_num = " << num << " in_channels = " << chin \
        << " img_h = " << hin << " img_w = " << win << " group = " << group \
        << " pad_width = " << pw << " pad_height = " << ph << " stride_width = " \
        << sw << " stride_height = " << sh << " dilation_w = " << dw \
        << " dilation_h = " << dh << " kernel_w = " << kw << " kernel_h = " \
        << kh << " out_channels = " << chout;
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    bool flag_relu = false;
    bool flag_bias = param.bias()->size() > 0;
    if (param.activation_param.has_active){
        if (param.activation_param.active == Active_relu){
            flag_relu = true;
        }
    }
    int group_size_in = win * hin * chin / group;
    int group_size_out = wout * hout * chout / group;
    int group_size_coldata = _m * _n;
    int hblock = get_hblock_int8(this->_ctx->get_arch());
    int m_roundup = hblock * ((_m + hblock - 1) / hblock);
    int group_size_weights = ((m_roundup * _k + 15) / 16) * 16;

    bool flag_1x1s1p1 = (_kw == 1) && (_kh == 1) && (param.stride_h == 1) && \
        (param.stride_w == 1) && (param.pad_w == 0) && (param.pad_h == 0) && \
        (param.dilation_w == 1) && (param.dilation_h == 1);

    const void* din = nullptr;
    void* dout = nullptr;
    const int8_t* weights = static_cast<const int8_t*>(param.weight()->data());
    //! input dtype transfer
    if (inputs[0]->get_dtype() != AK_INT8) {
        _tmp_in.set_dtype(AK_INT8);
        _tmp_in.reshape(inputs[0]->valid_shape());
        trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], 1.f, {1.f});
        din = _tmp_in.data();
    } else {
        din = inputs[0]->data();
    }

    if (outputs[0]->get_dtype() == AK_INT32) {
        dout = outputs[0]->mutable_data();
    } else {
        dout = _tmp_out.mutable_data();
    }

    for (int i = 0; i < num; ++i) {
        const int8_t* din_batch =  static_cast<const int8_t*>(din) + i * chin * hin * win;
        int32_t* dout_batch = static_cast<int32_t*>(dout) + i * chout * hout * wout;

        int32_t* col_data = static_cast<int32_t*>(_ctx->get_work_space()) + _ctx->get_l2_cache_size() / sizeof(int32_t);
        if (flag_1x1s1p1) {
            col_data = dout_batch;
        }
        for (int g = 0; g < group; ++g) {
            const int8_t* din_group = din_batch + g * group_size_in;
            const int8_t* weights_group = weights + g * group_size_weights;
            int32_t* coldata_group = col_data + g * group_size_coldata;
            gemm_prepack_int8(weights_group, din_group, nullptr, coldata_group, _m, _n, _k, \
                false, /*false*/flag_relu && (!flag_bias), false, nullptr, this->_ctx);
        }
        if (!flag_1x1s1p1) {
            col2im(col_data, chout, hout, wout, _kh, _kw, param.pad_h, param.pad_w, \
            param.stride_h, param.stride_w, param.dilation_h, param.dilation_w, dout_batch);
        }
        //! add bias
        if (flag_bias){
            fill_bias_relu<int32_t>(dout_batch, static_cast<const int32_t*>(param.bias()->data()), \
                chout, wout * hout, flag_bias, flag_relu);
        }
    }

    //! output dtype transfer
    if (outputs[0]->get_dtype() == AK_INT8) {
        trans_tensor_dtype<ARM, AK_INT32, AK_INT8>(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], \
        outputs[0]->get_scale()[0], param.weight()->get_scale());
    } else if (outputs[0]->get_dtype() == AK_FLOAT) {
        trans_tensor_dtype<ARM, AK_INT32, AK_FLOAT>(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], \
            1.f, param.weight()->get_scale());
    } else if (outputs[0]->get_dtype() != AK_INT32) {
        LOG(ERROR) << "unsupported precision type!!";
        return SaberInvalidValue;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    float op_macs = 2.f * kw * kh * num * chout * wout * hout * chin / group;
    LOG(INFO) << "Deconvlution int8: " << this->_op_name.c_str() << ", time: " << ts << \
        ", GOPs: " << 1e-9f * op_macs << ", GOPS: " << 0.000001 * op_macs / ts;
    GOPS ops;
    //fixme
    ops.ops = op_macs;
    ops.ts = ts;
    OpTimer::add_timer("Cast", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberDeconv2D, ConvParam, ARM, AK_HALF);
}
} // namespace anakin
