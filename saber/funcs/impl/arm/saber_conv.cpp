#include "saber/funcs/impl/arm/saber_conv.h"
#ifdef USE_ARM_PLACE
#include "saber/funcs/impl/arm/impl/conv_arm_impl.h"

namespace anakin{

namespace saber{

template <>
SaberConv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::SaberConv2D() {
    _impl = nullptr;
    _workspace_fwd_sizes = 0;
    _is_trans_weights = false;
    _flag_relu = false;
    _bias_term = true;
    _workspace_data = std::make_shared<Buffer<ARM>>();
    _weights_trans = std::make_shared<Buffer<ARM>>();
}

template <>
SaberConv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::~SaberConv2D() {}

template <>
SaberStatus SaberConv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::create(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs,\
    ConvParam<OpTensor> &conv_param, Context<ARM> &ctx) {

    this->_ctx = ctx;

    int threads = this->_ctx.get_act_ids().size();

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();

    _kw = conv_param.weight()->width();
    _kh = conv_param.weight()->height();

    int l1_cache = this->_ctx.devs[this->_ctx.get_device_id()]._info._L1_cache;
    int l2_cache = this->_ctx.devs[this->_ctx.get_device_id()]._info._L2_cache;
    //! if L1 cache size is not provided, set to 31K
    l1_cache = l1_cache > 0? l1_cache : 31000;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 2000000;

    if (conv_param.bias()->valid_size() > 0) {
        _bias_term = true;
    } else {
        _bias_term = false;
    }

    //! return basic conv func
    if (conv_param.dilation_h != 1 || conv_param.dilation_w != 1 \
        || conv_param.stride_h != conv_param.stride_w \
        || conv_param.stride_w > 2 || _kw != _kh || _kw > 3 \
        || ((conv_param.group > 1) && (conv_param.group != chin || conv_param.group != chout))) {
        //! basic conv
        _impl = conv_arm_basic;
        //LOG(ERROR) << "USE BASIC";
        return SaberSuccess;
    } else {
        //! depthwise conv, 3x3s1 or 3x3s2, pad must = 1
        if (conv_param.group == chin && chin == chout && _kw == 3 && conv_param.pad_w == 1 && \
                conv_param.pad_h == 1) {
            _impl = conv_depthwise_3x3;
            //LOG(ERROR) << "USE DW";
            return SaberSuccess;
        }

        //! for conv3x3s2 and input channel < 10, use gemm conv
        if (_kw < 3 || (_kw == 3 && (conv_param.stride_w == 2) || chin < 10)) {

            const int m = chout;
            const int n = hout * wout;
            const int k = chin * _kh * _kw;
            if (_kw == 1 && conv_param.stride_w == 1 && conv_param.pad_w == 0) {
                //! 1x1s1p0
                _impl = conv1x1s1_gemm;
                _workspace_fwd_sizes = 0;
            } else {
                //! otherwise
                _impl = conv_im2col_gemm;
                _workspace_fwd_sizes = sizeof(float) * k * n;
                _workspace_data->re_alloc(_workspace_fwd_sizes);
            }
            _gemmer.init(l1_cache, l2_cache, m, n, k, false, false, threads);
            //LOG(ERROR) << "USE GEMM";
            return SaberSuccess;
        }

        //! 3x3s1, input channel >= 10, use winograd
        //! if chin < 10, use sgemm, faster than winograd
        if (_kw == 3 && conv_param.stride_h == 1 && conv_param.pad_w == 1) {
            if (chout / (wout * hout) < 1) {
                //! use winograd
                _weights_trans->re_alloc(sizeof(float) * 8 * 8 * chout * chin * 2);
                //! space for computation
                int tile_w = (wout + 5) / 6;
                int tile_h = (hout + 5) / 6;
                int size_tile = tile_h * tile_w;
                int size_trans_channel = 8 * 8 * size_tile;
                int max_ch = chin > chout? chin : chout;

                //LOG(INFO) << "threads " << threads;

                _workspace_data->re_alloc(sizeof(float) * size_trans_channel * max_ch * 2);

                void* trans_tmp_ptr =(void*)((char*)_weights_trans->get_data_mutable() + \
                    sizeof(float) * 8 * 8 * chout * chin);
                float* weights_trans = (float*)_weights_trans->get_data_mutable();
                winograd_transform_weights(weights_trans, \
                    conv_param.weight()->data(), chout, chin, trans_tmp_ptr);

                //LOG(INFO) << "weighs size: " << this->_weight_data.size() << ", chout: " << chout << ", chin: " << chin;

                const int m_wino = chout;
                const int n_wino = size_tile;
                const int k_wino = chin;

                //LOG(INFO) << "threads " << threads << ", m " << m_wino << ", n " << n_wino << ", k " << k_wino;
                _gemmer.init(l1_cache, l2_cache, m_wino, n_wino, k_wino, false, false, threads);
                _impl = conv_arm_winograd3x3;
                _is_trans_weights = true;
                //LOG(ERROR) << "USE WINO";
            } else {
                //! use direct
                _impl = conv_3x3s1_direct;
                //LOG(ERROR) << "USE direct";
            }
            return SaberSuccess;
        }

    }

    _impl = conv_arm_basic;
    return SaberSuccess;
}

template <>
SaberStatus SaberConv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::init(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvParam<OpTensor> &conv_param, Context<ARM> &ctx) {
    return create(inputs, outputs, conv_param, ctx);
}

template <>
SaberStatus SaberConv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, ConvParam<OpTensor> &conv_param) {

    const float* weight = conv_param.weight()->data();
    if (_is_trans_weights) {
        weight = (float*)_weights_trans->get_data_mutable();
    }
    const float* bias = nullptr;
    if (_bias_term) {
        bias = conv_param.bias()->data();
    }
    _impl(*inputs[0], *outputs[0], weight, bias, conv_param.group, _kw, _kh, \
            conv_param.stride_w, conv_param.stride_h, conv_param.dilation_w, \
            conv_param.dilation_h, conv_param.pad_w, conv_param.pad_h, \
            _bias_term, _flag_relu, _gemmer, _workspace_data->get_data_mutable());
    return SaberSuccess;
}

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


