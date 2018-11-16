#include "saber/lite/funcs/saber_conv.h"
#ifdef USE_ARM_PLACE
#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#include "saber/lite/net/saber_factory_lite.h"
#include "saber/lite/funcs/neon/impl/sgemm_conv.h"
#include "saber/lite/core/tensor_op_lite.h"
namespace anakin{

namespace saber{

namespace lite{

SaberConv2D::SaberConv2D() {
    _impl = nullptr;
    _is_trans_weights = false;
    _param = nullptr;
    _act_funcs = nullptr;
}

SaberConv2D::SaberConv2D(ParamBase *param) {
    _param = (Conv2DParam*)param;
    this->_flag_param = true;
}

SaberConv2D::~SaberConv2D() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
    if (_act_funcs) {
        delete _act_funcs;
        _act_funcs = nullptr;
    }
    if (_act_param) {
        delete _act_param;
        _act_param = nullptr;
    }
}

SaberStatus SaberConv2D::load_param(ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (Conv2DParam*)param;
    if (_param->_flag_act && _param->_act_type != Active_relu) {
        if (_act_funcs == nullptr) {
            _act_funcs = new SaberActivation;
        }
        if (_act_param == nullptr) {
            _act_param = new ActivationParam;
        }
        ActivationParam act_param(_param->_act_type, _param->_neg_slope, \
            _param->_act_coef, _param->_act_channel_shared, \
            static_cast<const float*>(_param->_act_weights.data()), _param->_num_output);
        *_act_param = act_param;
        _act_funcs->load_param(_act_param);
    }
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberConv2D::load_param(std::istream &stream, const float *weights) {
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
        pad_w >> pad_h >> dila_w >> dila_h >> flag_bias >> w_offset >> b_offset >> flag_eltwise >> \
        flag_act >> act_type_i >> neg_slop >> act_coef >> act_channel_shared >> act_offset;

    ActiveType act_type = (ActiveType)act_type_i;

    _param = new Conv2DParam(weights_size, num_out, group, kw, kh, \
        stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias > 0, \
        weights + w_offset, weights + b_offset, flag_eltwise > 0, \
        flag_act, act_type, neg_slop, act_coef, act_channel_shared > 0, weights + act_offset);

    if (_param->_flag_act && act_type != Active_relu) {
        if (_act_funcs == nullptr) {
            _act_funcs = new SaberActivation;
        }
        if (_act_param == nullptr) {
            _act_param = new ActivationParam;
        }
        ActivationParam act_param(_param->_act_type, _param->_neg_slope, \
            _param->_act_coef, _param->_act_channel_shared, \
            static_cast<const float*>(_param->_act_weights.data()), _param->_num_output);
        *_act_param = act_param;
        _act_funcs->load_param(_act_param);
    }

    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberConv2D::set_op_precision(DataType ptype) {

    if (ptype == AK_INT8) {
        if (_param->_flag_act && (_param->_act_type != Active_relu)) {
            return SaberUnImplError;
        }
    }
    if (ptype == AK_FLOAT || ptype == AK_INT8) {
        _precision_type = ptype;
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
}

SaberStatus SaberConv2D::compute_output_shape(const std::vector<Tensor<CPU> *> &inputs,
                                              std::vector<Tensor<CPU> *> &outputs) {

    if (!this->_flag_param) {
        LOGE("ERROR: load conv param first\n");
        return SaberNotInitialized;
    }

    Shape output_shape = inputs[0]->valid_shape();
    LCHECK_EQ(inputs[0]->valid_shape().dims(), 4, \
        "ERROR: using reshape2d to reshape a 1d conv?\n");

    output_shape.set_num(inputs[0]->num()); // N
    output_shape.set_channel(_param->_num_output); // K

    int input_dim = inputs[0]->height(); // P
    int kernel_exten = _param->_dila_h * (_param->_kh - 1) + 1;
    int output_dim = (input_dim + 2 * _param->_pad_h - kernel_exten) / _param->_stride_h + 1;

    output_shape.set_height(output_dim);

    input_dim = inputs[0]->width(); // Q
    kernel_exten = _param->_dila_w * (_param->_kw - 1) + 1;
    output_dim = (input_dim + 2 * _param->_pad_w - kernel_exten) / _param->_stride_w + 1;

    output_shape.set_width(output_dim);

    return outputs[0]->set_shape(output_shape);
}

//template <>
SaberStatus SaberConv2D::init(\
    const std::vector<Tensor<CPU> *>& inputs, \
    std::vector<Tensor<CPU> *>& outputs, Context &ctx) {

    if (!this->_flag_param) {
        LOGE("ERROR: load conv param first\n");
        return SaberNotInitialized;
    }
    this->_ctx = &ctx;

    _is_trans_weights = false;

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();

    LCHECK_EQ(chin % _param->_group, 0, "ERROR: input channel or group size error\n");
    LCHECK_EQ(chout % _param->_group, 0, "ERROR: output channel or group size error\n");

    //! depthwise conv, 3x3s1 or 3x3s2, pad must = 1
    int k = 0;
    int m = 0;
    int n = 0;
    switch (this->get_op_precision()) {
        //! int8 init kernel
        case AK_INT8:
        {
            //! init int32 output
            _tmp_out.set_dtype(AK_INT32);
            _tmp_out.reshape(outputs[0]->valid_shape());
            //! trans bias to int32 for all implementation
            if (_param->_bias_term) {
                get_tensor_scale(_param->_weights, _w_scale, 0, 63.f);
                if (inputs[0]->get_scale().size() < 1) {
                    return SaberInvalidValue;
                }
                trans_fp32_bias_to_int32_inplace(_param->_bias, inputs[0]->get_scale()[0], _w_scale, _ctx);
            }

            if (_param->_group == chin && chin == chout && _param->_kw == 3 && _param->_kh == 3 && \
                    _param->_pad_w == 1 && _param->_pad_h == 1 && _param->_dila_w == 1 && _param->_dila_h == 1) {
                _impl = conv_depthwise_3x3_int8;
                trans_fp32_weights_to_int8_inplace(_param->_weights, 63.f, 0, _ctx);
                _w_scale = _param->_weights.get_scale();
                _is_trans_weights = false;
#if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                printf("%s USE INT8 DW, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
#endif
#ifdef ENABLE_OP_TIMER
                _conv_type = "conv_dw_int8";
#endif
                _flag_init = true;
                return SaberSuccess;
            }
#ifdef __aarch64__
            if (_param->_kw == 3 && _param->_kh == 3 && _param->_stride_h == 2 && _param->_stride_w == 2 && \
                    _param->_dila_w == 1 && _param->_dila_h == 1 && _param->_group == 1 && \
                    inputs[0]->width() <= 512 && inputs[0]->height() <= 512 && inputs[0]->channel() <= 16) {
                    // _weights_trans.reshape(Shape(((chout + 3) / 4) * 4, chin, 3, 3));
                    // Shape shape_out(((chout + 3) / 4) * 4, chin, 3, 3);
                    // _weights_trans.re_alloc(shape_out, AK_INT32);
                    // _is_trans_weights = true;
                    // conv3x3s2_trans_weights4c_int8(_weights_trans.mutable_data(), _param->_weights, chout, chin);
                    _impl = conv_3x3s2_direct_int8;
                    this->_flag_init = true;
                    trans_fp32_weights_to_int8(_param->_weights, _weights_trans, 63.f, 0, _ctx);
                    _w_scale = _weights_trans.get_scale();
                    _is_trans_weights = true;

//                    int wout_round = ((wout + 7) / 8) * 8;
//                    int win_round = wout_round * _param->_stride_w + 1;
//                    int tmp_size_out = wout_round * 1 * 4;
//                    int in_len = win_round * chin;
//                    int tmp_size_in = 3 * in_len;
//                    _ctx->workspace_extend(Shape(_ctx->get_threads() * tmp_size_out + tmp_size_in + wout_round + win_round));
            #if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                    printf("%s USE 3x3s2 int8 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
            #endif
            #ifdef ENABLE_OP_TIMER
                    _conv_type = "conv3x3s2_int8";
            #endif
                    return SaberSuccess;
                }
#else
            if (_param->_kw == 3 && _param->_kh == 3 && _param->_stride_h == 2 && _param->_stride_w == 2 && \
                    _param->_pad_w == 1 && _param->_pad_h == 1 && \
                    _param->_dila_w == 1 && _param->_dila_h == 1 && _param->_group == 1 && \
                    inputs[0]->width() > 7 && inputs[0]->height() > 7 && inputs[0]->width() <= 224 && \
                    inputs[0]->height() <= 224 && inputs[0]->channel() <= 16) {
                _impl = conv_3x3s2_direct_int8;
                trans_fp32_weights_to_int8(_param->_weights, _weights_trans, 63.f, 0,  _ctx);
                _w_scale = _weights_trans.get_scale();
                _is_trans_weights = true;
#if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                printf("%s USE 3x3s2 int8 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
#endif
#ifdef ENABLE_OP_TIMER
                _conv_type = "conv3x3s2_int8";
#endif
                _flag_init = true;
                return SaberSuccess;
            }
#endif //aarch64

            //! 3x3s1, when channel size or image size is large enough, use winograd
            //! otherwise use direct conv
            if (_param->_kw == 3 && _param->_kh == 3 && _param->_stride_h == 1 && _param->_stride_w == 1 && \
                    _param->_dila_w == 1 && _param->_dila_h == 1 && _param->_group == 1 && \
                    outputs[0]->channel() > 1) {

                if (inputs[0]->width() > 4 && inputs[0]->height() > 4 && \
                        _param->_pad_w == 1 && _param->_pad_h == 1 && (chin < 16 || chout < 14)) {
                    //! use direct
                    _impl = conv_3x3s1_direct_int8;
                    trans_fp32_weights_to_int8(_param->_weights, _weights_trans, 63.f, 0, _ctx);
                    _w_scale = _weights_trans.get_scale();
                    _is_trans_weights = true;
#if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                    printf("%s USE 3x3s1 int8 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                            this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
#endif
#ifdef ENABLE_OP_TIMER
                    _conv_type = "conv3x3s1_dir_int8";
#endif
                    _flag_init = true;
                    return SaberSuccess;
                }
            }

            //! use im2col and int8 gemm conv
            m = chout / _param->_group;
            n = hout * wout;
            k = chin * _param->_kh * _param->_kw / _param->_group;
            if (_param->_kw == 1 && _param->_kh == 1 && _param->_stride_w == 1 && _param->_stride_h == 1 && \
                        _param->_pad_w == 0 && _param->_pad_h == 0) {
                //! 1x1s1p0
                _impl = conv1x1s1_gemm_int8;
#ifdef ENABLE_OP_TIMER
                _conv_type = "conv1x1_int8";
#endif
                if (n > 1) {
                    trans_fp32_weights_to_int8_inplace_gemm(_param->_weights, 63.f, false, _param->_group, _ctx);
                    _w_scale = _param->_weights.get_scale();
                    _is_trans_weights = false;
                }
#if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                printf("%s USE GEMM_INT8, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
#endif
                _flag_init = true;
                return SaberSuccess;
            } else {
                //! otherwise
                if (_param->_kh == 3 && _param->_kw == 3 && _param->_stride_h == 1 && _param->_stride_w == 1 && n > 1) {
                    _idx_data.reshape(Shape(n * _param->_kh * _param->_kw));
                    int* idx_out = (int*)_idx_data.mutable_data();
                    for (int i = 0; i < hout; ++i) {
                        for (int j = 0; j < wout; ++j) {
                            compute_offset(idx_out, i, j, _param->_kh, _param->_kw, hin, win, _param->_pad_h, _param->_pad_w, _param->_dila_h, _param->_dila_w);
                            idx_out += _param->_kh * _param->_kw;
                        }
                    }
                }
                _impl = conv_im2col_gemm_int8;
                //_workspace_data.reshape(Shape(k * n));
                _ctx->workspace_extend(Shape(k * n + n));
#ifdef ENABLE_OP_TIMER
                std::ostringstream ss;
                    ss << "conv" << _param->_kh << "x" << _param->_kw << "_s" << _param->_stride_w << "_p" \
                        << _param->_pad_w << "_d" << _param->_dila_w << "_g" << _param->_group;
                    _conv_type = ss.str();
#endif
                if (n > 1) {
                    trans_fp32_weights_to_int8_gemm(_param->_weights, _weights_trans, 63.f, false, _param->_group, _ctx);
                    _w_scale = _weights_trans.get_scale();
                    _is_trans_weights = true;
                }
#if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                printf("%s USE GEMM_INT8, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
#endif
                _flag_init = true;
                return SaberSuccess;
            }

        }
            break;
        case AK_FLOAT:
            if (_param->_group == chin && chin == chout && _param->_kw == 3 && _param->_kh == 3 && \
                    _param->_pad_w == 1 && _param->_pad_h == 1 && _param->_dila_w == 1 && _param->_dila_h == 1) {
                    _impl = conv_depthwise_3x3;
            #if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                    printf("%s USE DW, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
            #endif
                    this->_flag_init = true;
            #ifdef ENABLE_OP_TIMER
                   _conv_type = "conv_dw";
            #endif
                    return SaberSuccess;
                }
            #if 1 //conv_1x5
                if (_param->_group == 1 && _param->_kw == 5 && _param->_kh == 1 && _param->_dila_w == 1 && _param->_dila_h == 1 && \
                    _param->_pad_w == 1 && _param->_num_output % 2 == 0) {
                    _impl = conv_1x5s1_direct;
            #if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                    printf("%s USE 1x5 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
            #endif
                    this->_flag_init = true;
            #ifdef ENABLE_OP_TIMER
                    _conv_type = "conv1x5s1";
            #endif
                    return SaberSuccess;
                }
            #endif //conv_1x5

            #if 1 //conv_5x1
            if (_param->_group == 1 && _param->_kw == 1 && _param->_kh == 5 && _param->_dila_w == 1 && _param->_dila_h == 1) {
                _impl = conv_5x1s1_direct;
            #if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                printf("%s USE 5x1 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
            #endif
                this->_flag_init = true;
            #ifdef ENABLE_OP_TIMER
                _conv_type = "conv5x1s1";
            #endif
                return SaberSuccess;
            }
            #endif //conv_5x1

            #if 1//! 3x3s2p1, direct
            #ifdef __aarch64__
                if (_param->_kw == 3 && _param->_kh == 3 && _param->_stride_h == 2 && _param->_stride_w == 2 && \
                    _param->_dila_w == 1 && _param->_dila_h == 1 && _param->_group == 1 && \
                    inputs[0]->width() <= 512 && inputs[0]->height() <= 512 && inputs[0]->channel() <= 16) {
                    // _weights_trans.reshape(Shape(((chout + 3) / 4) * 4, chin, 3, 3));
                    Shape shape_out(((chout + 3) / 4) * 4, chin, 3, 3);
                    _weights_trans.re_alloc(shape_out, AK_FLOAT);
                    _is_trans_weights = true;
                    conv3x3s2_trans_weights4c(_weights_trans.mutable_data(), _param->_weights.data(), chout, chin);
                    _impl = conv_3x3s2_direct;
                    this->_flag_init = true;
                    int wout_round = ((wout + 3) / 4) * 4;
                    int win_round = wout_round * _param->_stride_w + 1;
                    int tmp_size_out = wout_round * 2 * 4;
                    int in_len = win_round * chin;
                    int tmp_size_in = 5 * in_len;
                    _ctx->workspace_extend(Shape(_ctx->get_threads() * tmp_size_out + tmp_size_in + wout_round + win_round));
            #if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                    printf("%s USE 3x3s2 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
            #endif
            #ifdef ENABLE_OP_TIMER
                    _conv_type = "conv3x3s2";
            #endif
                    return SaberSuccess;
                }
            #else
                if (_param->_kw == 3 && _param->_kh == 3 && _param->_stride_h == 2 && _param->_stride_w == 2 && \
                    _param->_pad_w == 1 && _param->_pad_h == 1 && \
                    _param->_dila_w == 1 && _param->_dila_h == 1 && _param->_group == 1 && \
                    inputs[0]->width() > 7 && inputs[0]->height() > 7 && inputs[0]->width() <= 224 && \
                    inputs[0]->height() <= 224 && inputs[0]->channel() <= 16) {
                    _impl = conv_3x3s2_direct;
                    this->_flag_init = true;
            #if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                    printf("%s USE 3x3s2 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
            #endif
            #ifdef ENABLE_OP_TIMER
                    _conv_type = "conv3x3s2";
            #endif
                    return SaberSuccess;
                }
            #endif //aarch64

            #endif //if conv3x3s2

            #if 1
                //! 3x3s1, when channel size or image size is large enough, use winograd
                //! otherwise use direct conv
                if (_param->_kw == 3 && _param->_kh == 3 && _param->_stride_h == 1 && _param->_stride_w == 1 && \
                    _param->_dila_w == 1 && _param->_dila_h == 1 && _param->_group == 1 && \
                    outputs[0]->channel() > 1) {

                    if (inputs[0]->width() > 4 && inputs[0]->height() > 4 && \
                        _param->_pad_w == 1 && _param->_pad_h == 1 && (chin < 16 || chout < 14)) {
                        //! use direct
                        _impl = conv_3x3s1_direct;
            #if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                        printf("%s USE 3x3s1 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                            this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
            #endif
            #ifdef ENABLE_OP_TIMER
                        _conv_type = "conv3x3_dir";
            #endif
                    } else {
                        //! use winograd
                        //! space for computation
                        int tile_w = (wout + 5) / 6;
                        int tile_h = (hout + 5) / 6;
                        int size_tile = tile_h * tile_w;
                        int size_trans_channel = 8 * 8 * size_tile;
                        int max_ch = chin > chout? chin : chout;

                        const int m_wino = chout;
                        const int n_wino = size_tile;
                        const int k_wino = chin;

                        int hblock = get_hblock(this->_ctx->get_arch());

                        int m_round = hblock * ((m_wino + hblock - 1) / hblock);
                        // _weights_trans.reshape(Shape(8 * 8 * m_round * chin)); //for prepack gemm
                        Shape shape_out(8 * 8 * m_round * chin);
                        _weights_trans.re_alloc(shape_out, AK_FLOAT);
                        //_workspace_data.reshape(Shape(size_trans_channel * max_ch * 2)); // workspace for trans input and output
                        _ctx->workspace_extend(Shape(size_trans_channel * max_ch * 2 + n_wino));

                        float* weights_wino = static_cast<float*>(fast_malloc(sizeof(float) * 8 * 8 * chout * chin));

                        void* trans_tmp_ptr = fast_malloc(sizeof(float) * 8 * 8 * chout * chin);

                        winograd_transform_weights(weights_wino, _param->_weights.data(), chout, chin, trans_tmp_ptr);
                        fast_free(trans_tmp_ptr);

                        float* weights_trans = static_cast<float*>(_weights_trans.mutable_data());
                        for (int i = 0; i < 64; ++i) {
                            float* packed_weights = weights_trans + i * m_round * chin;
                            const float* weights_wino_ptr = weights_wino + i * chout * chin;
                            prepackA(packed_weights, weights_wino_ptr, chin, 0, m_wino, 0, chin, false, this->_ctx);
                        }
                        fast_free(weights_wino);
                        _impl = conv_arm_winograd3x3;
                        _is_trans_weights = true;
            #if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                        printf("%s USE WINOGRAD, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                            this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
            #endif
            #ifdef ENABLE_OP_TIMER
                        _conv_type = "conv3x3_wino";
            #endif
                    }

                    if (_act_funcs) {
                        _act_funcs->init(outputs, outputs, ctx);
                    }
                    this->_flag_init = true;
                    return SaberSuccess;
                }
            #endif

                //! use im2col and gemm conv
                m = chout / _param->_group;
                n = hout * wout;
                k = chin * _param->_kh * _param->_kw / _param->_group;
                if (_param->_kw == 1 && _param->_kh == 1 && _param->_stride_w == 1 && _param->_stride_h == 1 && \
                        _param->_pad_w == 0 && _param->_pad_h == 0) {
                    //! 1x1s1p0
                    _impl = conv1x1s1_gemm;
                    _ctx->workspace_extend(Shape(n));
            #ifdef ENABLE_OP_TIMER
                    _conv_type = "conv1x1";
            #endif
                } else {
                    //! otherwise
                    if (_param->_kh == 3 && _param->_kw == 3 && _param->_stride_h == 1 && _param->_stride_w == 1 && n > 1) {
                        _idx_data.reshape(Shape(n * _param->_kh * _param->_kw));
                        int* idx_out = (int*)_idx_data.mutable_data();
                        for (int i = 0; i < hout; ++i) {
                            for (int j = 0; j < wout; ++j) {
                                compute_offset(idx_out, i, j, _param->_kh, _param->_kw, hin, win, _param->_pad_h, _param->_pad_w, _param->_dila_h, _param->_dila_w);
                                idx_out += _param->_kh * _param->_kw;
                            }
                        }
                    }
                    _impl = conv_im2col_gemm;
                    //_workspace_data.reshape(Shape(k * n));
                    _ctx->workspace_extend(Shape(k * n));
            #ifdef ENABLE_OP_TIMER
                    std::ostringstream ss;
                    ss << "conv" << _param->_kh << "x" << _param->_kw << "_s" << _param->_stride_w << "_p" \
                        << _param->_pad_w << "_d" << _param->_dila_w << "_g" << _param->_group;
                    _conv_type = ss.str();
            #endif
                }
                if (n > 1) {
                    int hblock = get_hblock(this->_ctx->get_arch());
                    int m_roundup = hblock * ((m + hblock - 1) / hblock);
                    int group_size_round_up = ((m_roundup * k + 15) / 16) * 16;
                    _weights_trans.reshape(Shape(group_size_round_up * _param->_group));
                    _is_trans_weights = true;
                    for (int g = 0; g < _param->_group; ++g) {
                        const float* weights_group = static_cast<const float*>(_param->_weights.data()) + g * m * k;
                        float* weights_trans_ptr = static_cast<float*>(_weights_trans.mutable_data()) + g * group_size_round_up;
                        prepackA(weights_trans_ptr, weights_group, k, 0, m, 0, k, false, this->_ctx);
                    }
                }
            #if defined(ENABLE_DEBUG) || defined(ENABLE_OP_TIMER)
                printf("%s USE GEMM, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                        this->get_op_name(), num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
            #endif
            break;
        default:
            LOGF("data type: %d is unsupported now", (int)(this->get_op_precision()));
        }
    this->_flag_init = true;
    return SaberSuccess;
}

//template <>
SaberStatus SaberConv2D::dispatch(const std::vector<Tensor<CPU> *>& inputs, \
    std::vector<Tensor<CPU> *>& outputs) {

    if (!this->_flag_init) {
        LOGE("ERROR: init conv first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif
    bool flag_act = _param->_flag_act && _param->_act_type == Active_relu;

    const void* weight = _param->_weights.data();
    if (_is_trans_weights) {
        weight = _weights_trans.data();
//        print_tensor(_weights_trans);
    }
    const void* bias = nullptr;
    if (_param->_bias_term) {
        bias = _param->_bias.data();
    }
    int num = inputs[0]->num();
    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();

    const void* din = nullptr;
    void* dout = nullptr;
/*
    DataType type = inputs[0]->get_dtype();
    if (type == AK_FLOAT){
        for (int j = 0; j < inputs.size(); j++){
            std::vector<float> scale_out;
            auto state = get_tensor_scale(*inputs[j], scale_out, 0, 63.f);
            inputs[j]->set_scale(scale_out);
            if (state != SaberSuccess){
                LOGE("----------------------------------------set_scale ERROR");
            }
            printf("input scale_out: \n");
            for (int i = 0; i < scale_out.size(); i++){
                printf("%.3f  ", scale_out[i]);
            }
            printf("\n");
        }
    }
*/
    if (this->get_op_precision() == AK_INT8) {
        if (inputs[0]->get_dtype() != AK_INT8) {
//            LOGI("conv int8 trans input, fp32 to int8\n");
            _tmp_in.set_dtype(AK_INT8);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_fp32_to_int8(*inputs[0], _tmp_in, _ctx);
            din = _tmp_in.data();
        } else {
//            LOGI("conv int8 trans input, no trans\n");
            din = inputs[0]->data();
        }
    } else if (this->get_op_precision() == AK_FLOAT) {
        if (inputs[0]->get_dtype() != AK_FLOAT) {
//            LOGI("conv fp32 trans input, int8 to fp32\n");
            _tmp_in.set_dtype(AK_FLOAT);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_int8_to_fp32(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], _ctx);
            din = _tmp_in.data();
        } else {
//            LOGI("conv fp32 trans input, no trans\n");
            din = inputs[0]->data();
        }
    } else {
        LOGE("ERROR: unsupported input data type!!\n");
        return SaberInvalidValue;
    }

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

    _impl(din, dout, num, chout, hout, wout, \
            chin, hin, win, weight, bias, _param->_group, _param->_kw, _param->_kh, _param->_stride_w, _param->_stride_h, \
            _param->_dila_w, _param->_dila_h, _param->_pad_w, _param->_pad_h, _param->_bias_term, flag_act, this->_ctx, \
            nullptr, (const void*)_idx_data.data());

    if (_act_funcs) {
        _act_funcs->dispatch(outputs, outputs);
    }

    if (this->get_op_precision() == AK_INT8) {
        if (outputs[0]->get_dtype() == AK_INT8) {
//            LOGI("conv trans output, int32 to int8\n");
            trans_tensor_int32_to_int8(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], _w_scale, _ctx);
        } else if (outputs[0]->get_dtype() == AK_FLOAT) {
//            LOGI("conv trans output, int32 to fp32\n");
            trans_tensor_int32_to_fp32(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], _w_scale, _ctx);
        } else {
            LOGE("unsupported precision type!!\n");
            return SaberInvalidValue;
        }
    }
#ifdef ENABLE_OP_TIMER
    float op_macs = _param->_kw * _param->_kh * \
        num * chout * wout * hout * chin / _param->_group;
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    LOGI("type: %s, name: %s, conv time: %f ms, %f GOPs, %f GOPS\n", _conv_type.c_str(), this->get_op_name(), ts, 1e-9f * op_macs, 0.000001 * op_macs / ts);
    GOPS ops;
    ops.ts = ts;
    ops.ops = op_macs;
    OpTimer::add_timer("convolution", ops);
    OpTimer::add_timer("total", ops);
    OpTimer::add_timer(_conv_type, ops);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberConv2D);
} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


