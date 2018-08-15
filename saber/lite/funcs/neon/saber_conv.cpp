#include "saber/lite/funcs/saber_conv.h"
#ifdef USE_ARM_PLACE
#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"

namespace anakin{

namespace saber{

namespace lite{

SaberConv2D::SaberConv2D() {
    _impl = nullptr;
    _workspace_fwd_sizes = 0;
    _is_trans_weights = false;
    _flag_relu = false;
    _bias_term = true;
}

SaberConv2D::SaberConv2D(int weights_size, int num_output, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, \
        bool flag_bias, const float* weights, const float* bias) {
    _num_output = num_output;
    _group = group;
    _kw = kw;
    _kh = kh;
    _stride_w = stride_w;
    _stride_h = stride_h;
    _pad_w = pad_w;
    _pad_h = pad_h;
    _dila_w = dila_w;
    _dila_h = dila_h;
    _bias_term = flag_bias;
    _weights = weights;
    _bias = bias;
    _weights_size = weights_size;
}

SaberStatus SaberConv2D::load_param(int weights_size, int num_output, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, \
        bool flag_bias, const float* weights, const float* bias) {
    _num_output = num_output;
    _group = group;
    _kw = kw;
    _kh = kh;
    _stride_w = stride_w;
    _stride_h = stride_h;
    _pad_w = pad_w;
    _pad_h = pad_h;
    _dila_w = dila_w;
    _dila_h = dila_h;
    _bias_term = flag_bias;
    _weights = weights;
    _bias = bias;
    _weights_size = weights_size;
    return SaberSuccess;
}

SaberStatus SaberConv2D::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                              std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    Shape output_shape = inputs[0]->valid_shape();
    LCHECK_EQ(inputs[0]->valid_shape().dims(), 4, "using reshape2d to reshape a 1d conv?");

    output_shape.set_num(inputs[0]->num()); // N
    output_shape.set_channel(_num_output); // K

    int input_dim = inputs[0]->height(); // P
    int kernel_exten = _dila_h * (_kh - 1) + 1;
    int output_dim = (input_dim + 2 * _pad_h - kernel_exten) / _stride_h + 1;

    output_shape.set_height(output_dim);

    input_dim = inputs[0]->width(); // Q
    kernel_exten = _dila_w * (_kw - 1) + 1;
    output_dim = (input_dim + 2 * _pad_w - kernel_exten) / _stride_w + 1;

    output_shape.set_width(output_dim);

    return outputs[0]->set_shape(output_shape);
}

//template <>
SaberStatus SaberConv2D::init(\
    const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs, \
    std::vector<Tensor<CPU, AK_FLOAT> *>& outputs, Context &ctx) {

    _ctx = ctx;

    int threads = _ctx.get_act_ids().size();

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();

    int l1_cache = Env::cur_env()._L1_cache;
    int l2_cache = Env::cur_env()._L2_cache;
    //! if L1 cache size is not provided, set to 31K
    l1_cache = l1_cache > 0? l1_cache : 31000;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 2000000;

    LCHECK_EQ(chin % _group, 0, "input channel or group size error");
    LCHECK_EQ(chout % _group, 0, "output channel or group size error");
#if 0
    //! return basic conv func
    if (_dila_h != 1 || _dila_w != 1) {
        //! basic conv
        _impl = conv_arm_basic;
        printf("USE BASIC\n");
        return SaberSuccess;
    }
#endif
    //! depthwise conv, 3x3s1 or 3x3s2, pad must = 1
    if (_group == chin && chin == chout && _kw == 3 && _pad_w == 1 && _pad_h == 1) {
        _impl = conv_depthwise_3x3;
        printf("USE DW\n");
        return SaberSuccess;
    }

    //! 3x3s1, when channel size or image size is large enough, use winograd
    //! otherwise use direct conv

    if (_kw == 3 && _kh == 3 && _stride_h == 1 && \
        _pad_w == 1 && _group == 1) {

        if (chout / (wout * hout) > 1 || chin < 16 || chout < 14) {
            //! use direct
            _impl = conv_3x3s1_direct;
            printf("USE 3x3 direct\n");
        } else {
            //! use winograd
            _weights_trans.reshape(Shape(8 * 8 * chout * chin * 2));
            //! space for computation
            int tile_w = (wout + 5) / 6;
            int tile_h = (hout + 5) / 6;
            int size_tile = tile_h * tile_w;
            int size_trans_channel = 8 * 8 * size_tile;
            int max_ch = chin > chout? chin : chout;

            _workspace_data.reshape(Shape(size_trans_channel * max_ch * 2));

            void* trans_tmp_ptr =(void*)(_weights_trans.mutable_data() + 8 * 8 * chout * chin);
            float* weights_trans = _weights_trans.mutable_data();
            winograd_transform_weights(weights_trans, _weights, chout, chin, trans_tmp_ptr);

            const int m_wino = chout;
            const int n_wino = size_tile;
            const int k_wino = chin;

            _gemmer.init(l1_cache, l2_cache, m_wino, n_wino, k_wino, false, false, threads);
            _impl = conv_arm_winograd3x3;
            _is_trans_weights = true;
            printf("USE WINO\n");
        }
        return SaberSuccess;
    }

        //! use im2col and gemm conv
        const int m = chout / _group;
        const int n = hout * wout;
        const int k = chin * _kh * _kw / _group;
        if (_kw == 1 && _kh == 1 && _stride_w == 1 && _stride_h == 1 && \
            _pad_w == 0 && _pad_h == 0) {
            //! 1x1s1p0
            _impl = conv1x1s1_gemm;
            _workspace_fwd_sizes = 0;
        } else {
            //! otherwise
            _impl = conv_im2col_gemm;
            _workspace_fwd_sizes = k * n;
            _workspace_data.reshape(Shape(_workspace_fwd_sizes));
        }

        _gemmer.init(l1_cache, l2_cache, m, n, k, false, false, threads);
        printf("USE GEMM\n");
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

    const float* weight = _weights;
    if (_is_trans_weights) {
        weight = _weights_trans.data();
    }
    const float* bias = nullptr;
    if (_bias_term) {
        bias = _bias;
    }
    int num = inputs[0]->num();
    int chout = outputs[0]->channel();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    _impl(inputs[0]->data(), outputs[0]->mutable_data(), num, chout, hout, wout, \
            chin, hin, win, weight, bias, _group, _kw, _kh, _stride_w, _stride_h, \
            _dila_w, _dila_h, _pad_w, _pad_h, _bias_term, _flag_relu, _gemmer, \
            (void*)_workspace_data.mutable_data());
    return SaberSuccess;
}

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


