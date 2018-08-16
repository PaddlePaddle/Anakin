#include "saber/lite/funcs/saber_conv.h"
#ifdef USE_ARM_PLACE
#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#include "saber/lite/net/saber_factory_lite.h"
namespace anakin{

namespace saber{

namespace lite{

SaberConv2D::SaberConv2D() {
    _impl = nullptr;
    _workspace_fwd_sizes = 0;
    _is_trans_weights = false;
    _flag_relu = false;
    //_bias_term = true;
}

SaberConv2D::SaberConv2D(const ParamBase *param) {
    _param = (const Conv2DParam*)param;
    this->_flag_param = true;
}

SaberConv2D::~SaberConv2D() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberConv2D::load_param(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const Conv2DParam*)param;
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

    stream >> weights_size >> num_out >> group >> kw >> kh >> stride_w >> stride_h >> \
        pad_w >> pad_h >> dila_w >> dila_h >> flag_bias >> w_offset >> b_offset;
    _param = new Conv2DParam(weights_size, num_out, group, kw, kh, \
        stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias>0, \
        weights + w_offset, weights + b_offset);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#if 0
SaberStatus SaberConv2D::load_param(FILE *fp, const float* weights) {
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
    fscanf(fp, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
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
           &w_offset,
           &b_offset);
    _param = new Conv2DParam(weights_size, num_out, group, kw, kh, \
        stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias>0, \
        weights + w_offset, weights + b_offset);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#endif
SaberStatus SaberConv2D::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                              std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_param) {
        printf("load conv param first\n");
        return SaberNotInitialized;
    }

    Shape output_shape = inputs[0]->valid_shape();
    LCHECK_EQ(inputs[0]->valid_shape().dims(), 4, "using reshape2d to reshape a 1d conv?");

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

#ifdef ENABLE_OP_TIMER
    this->_op_macs = _param->_kw * _param->_kh * \
        output_shape.num() * output_shape.channel() * output_shape.width() * output_shape.height() * \
        inputs[0]->channel() / _param->_group;
#endif

    return outputs[0]->set_shape(output_shape);
}

//template <>
SaberStatus SaberConv2D::init(\
    const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs, \
    std::vector<Tensor<CPU, AK_FLOAT> *>& outputs, Context &ctx) {

    if (!this->_flag_param) {
        printf("load conv param first\n");
        return SaberNotInitialized;
    }
    this->_ctx = &ctx;
    //printf("conv init \n");

    int threads = this->_ctx->get_threads();

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();

    // printf("kw: %d, kh: %d\n", _kw, _kh);
    int l1_cache = Env::cur_env()._L1_cache;
    int l2_cache = Env::cur_env()._L2_cache;
    //! if L1 cache size is not provided, set to 31K
    l1_cache = l1_cache > 0? l1_cache : 31000;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 2000000;

    LCHECK_EQ(chin % _param->_group, 0, "input channel or group size error");
    LCHECK_EQ(chout % _param->_group, 0, "output channel or group size error");

    //! depthwise conv, 3x3s1 or 3x3s2, pad must = 1
    if (_param->_group == chin && chin == chout && _param->_kw == 3 && _param->_kh == 3 && \
            _param->_pad_w == 1 && _param->_pad_h == 1 && _param->_dila_w == 1 && _param->_dila_h == 1) {
        _impl = conv_depthwise_3x3;
        printf("USE DW, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
            num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
        this->_flag_init = true;
#ifdef ENABLE_OP_TIMER
       _conv_type = "conv_dw";
#endif
        return SaberSuccess;
    }
#if __aarch64__ //armv8 conv3x3s2 is not support now

#else
    //! 3x3s2p1, direct
    if (_param->_kw == 3 && _param->_kh == 3 && _param->_stride_h == 2 && _param->_stride_w == 2 && \
        _param->_pad_w == 1 && _param->_pad_h == 1 && _param->_dila_w == 1 && _param->_dila_h == 1 && _param->_group == 1 && \
        inputs[0]->width() > 7 && inputs[0]->height() > 7 && outputs[0]->channel() > 1) {
        _impl = conv_3x3s2_direct;
        this->_flag_init = true;
        printf("USE 3x3s2 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
            num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
#ifdef ENABLE_OP_TIMER
        _conv_type = "conv3x3s2";
#endif
        return SaberSuccess;
    }
#endif
    //! 3x3s1, when channel size or image size is large enough, use winograd
    //! otherwise use direct conv
    if (_param->_kw == 3 && _param->_kh == 3 && _param->_stride_h == 1 && _param->_stride_w == 1 && \
        _param->_pad_w == 1 && _param->_pad_h == 1 && _param->_dila_w == 1 && _param->_dila_h == 1 && _param->_group == 1 && \
        inputs[0]->width() > 4 && inputs[0]->height() > 4 && outputs[0]->channel() > 1) {

        if (chout / (wout * hout) > 1 || chin < 16 || chout < 14) {
            //! use direct
            _impl = conv_3x3s1_direct;
            printf("USE 3x3s1 direct, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
#ifdef ENABLE_OP_TIMER
            _conv_type = "conv3x3_dir";
#endif
        } else {
            //! use winograd
            _weights_trans.reshape(Shape(8 * 8 * chout * chin * 2));
            //! space for computation
            int tile_w = (wout + 5) / 6;
            int tile_h = (hout + 5) / 6;
            int size_tile = tile_h * tile_w;
            int size_trans_channel = 8 * 8 * size_tile;
            int max_ch = chin > chout? chin : chout;

            //LOG(INFO) << "threads " << threads;

            _workspace_data.reshape(Shape(size_trans_channel * max_ch * 2));

            void* trans_tmp_ptr =(void*)(_weights_trans.mutable_data() + 8 * 8 * chout * chin);
            float* weights_trans = _weights_trans.mutable_data();
            winograd_transform_weights(weights_trans, _param->_weights, chout, chin, trans_tmp_ptr);

            //LOG(INFO) << "weighs size: " << this->_weight_data.size() << ", chout: " << chout << ", chin: " << chin;

            const int m_wino = chout;
            const int n_wino = size_tile;
            const int k_wino = chin;

            //LOG(INFO) << "threads " << threads << ", m " << m_wino << ", n " << n_wino << ", k " << k_wino;
            _gemmer.init(l1_cache, l2_cache, m_wino, n_wino, k_wino, false, false, threads, this->_ctx->get_work_space());
            _impl = conv_arm_winograd3x3;
            _is_trans_weights = true;
            printf("USE WINOGRAD, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
                num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
#ifdef ENABLE_OP_TIMER
            _conv_type = "conv3x3_wino";
#endif
        }
        this->_flag_init = true;
        return SaberSuccess;
    }

    //! use im2col and gemm conv
    const int m = chout / _param->_group;
    const int n = hout * wout;
    const int k = chin * _param->_kh * _param->_kw / _param->_group;
    if (_param->_kw == 1 && _param->_kh == 1 && _param->_stride_w == 1 && _param->_stride_h == 1 && \
            _param->_pad_w == 0 && _param->_pad_h == 0) {
        //! 1x1s1p0
        _impl = conv1x1s1_gemm;
        _workspace_fwd_sizes = 0;
#ifdef ENABLE_OP_TIMER
        _conv_type = "conv1x1";
#endif
    } else {
        //! otherwise
        _impl = conv_im2col_gemm;
        _workspace_fwd_sizes = k * n;
        _workspace_data.reshape(Shape(_workspace_fwd_sizes));
#ifdef ENABLE_OP_TIMER
        std::ostringstream ss;
        ss << "conv" << _param->_kw << "x" << _param->_kh << "_s" << _param->_stride_w << "_p" \
            << _param->_pad_w << "_d" << _param->_dila_w << "_g" << _param->_group;
        _conv_type = ss.str();
#endif
    }

    _gemmer.init(l1_cache, l2_cache, m, n, k, false, false, threads, this->_ctx->get_work_space());
    printf("USE GEMM, num=%d, channel=%d, height=%d, width=%d, group=%d, kernel=%d, stride=%d, dila=%d, pad=%d\n", \
            num, chin, hin, win, _param->_group, _param->_kw, _param->_stride_w, _param->_dila_w, _param->_pad_w);
    this->_flag_init = true;
    return SaberSuccess;
}

SaberStatus SaberConv2D::set_activation(bool flag) {
    _flag_relu = flag;
    return SaberSuccess;
}


//template <>
SaberStatus SaberConv2D::dispatch(\
    const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs, \
    std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) {

    if (!this->_flag_init) {
        printf("init conv first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    const float* weight = _param->_weights;
    if (_is_trans_weights) {
        weight = _weights_trans.data();
    }
    const float* bias = nullptr;
    if (_param->_bias_term) {
        bias = _param->_bias;
    }
    int num = inputs[0]->num();
    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    _impl(inputs[0]->data(), outputs[0]->mutable_data(), num, chout, hout, wout, \
            chin, hin, win, weight, bias, _param->_group, _param->_kw, _param->_kh, _param->_stride_w, _param->_stride_h, \
            _param->_dila_w, _param->_dila_h, _param->_pad_w, _param->_pad_h, _param->_bias_term, _flag_relu, _gemmer, \
            (void*)_workspace_data.mutable_data());

#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("type: %s, name: %s, conv time: %f ms, %f GOPs, %f GOPS\n", _conv_type.c_str(), this->get_op_name(), ts, 1e-9f * this->_op_macs, 0.000001 * this->_op_macs / ts);
    OpTimer::add_timer("convolution", ts);
    OpTimer::add_timer("total", ts);
    OpTimer::add_timer(_conv_type, ts);
#endif
    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberConv2D);
} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


