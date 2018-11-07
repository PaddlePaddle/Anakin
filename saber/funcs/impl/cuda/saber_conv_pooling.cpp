
#include "saber/funcs/impl/cuda/saber_conv_pooling.h"
#include "sass_funcs.h"
#include "saber/funcs/calibrate.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberConv2DPooling<NV, AK_FLOAT>::\
create(const std::vector<Tensor<NV> *>& inputs,
       std::vector<Tensor<NV> *>& outputs,
       ConvPoolingParam<NV>& param, Context<NV>& ctx) {
    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);

    if (_use_k3p) {
        dispatch_func = winograd_conv_relu_pooling<float, float>;
    } else if (_use_kp) {
        const int K = param.conv_param.weight()->num();

        if (K % 4 == 0) {
            dispatch_func = direct_conv_bias_relu_maxpool2k2s0p_Kdivis4<float, float>;
        } else {
            dispatch_func = direct_conv_bias_relu_maxpool2k2s0p_Kindiv4<float, float>;
        }
    } else {
        _inner_tensor.reshape(_inner_shape);
        _inner_tensor_v.resize(1);
        _inner_tensor_v[0] = &_inner_tensor;
        _conv.create(inputs, _inner_tensor_v, param.conv_param, ctx);
        _pool.create(_inner_tensor_v, outputs, param.pooling_param, ctx);
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberConv2DPooling<NV, AK_FLOAT>::
init(const std::vector<Tensor<NV> *>& inputs,
     std::vector<Tensor<NV> *>& outputs,
     ConvPoolingParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _kernel_height = param.conv_param.weight()->height();
    _kernel_width = param.conv_param.weight()->width();

    _use_k3p = false; // enable winograd + pooling
    _use_kp = false; // enable direct + pooling

    _use_k3p = _use_k3p && (param.conv_param.weight()->height() == 3);
    _use_k3p = _use_k3p && (param.conv_param.weight()->width() == 3);
    _use_k3p = _use_k3p && (param.conv_param.stride_h == 1);
    _use_k3p = _use_k3p && (param.conv_param.stride_w == 1);
    _use_k3p = _use_k3p && (param.conv_param.dilation_h == 1);
    _use_k3p = _use_k3p && (param.conv_param.dilation_w == 1);
    _use_k3p = _use_k3p && (param.conv_param.group == 1);
    _use_k3p = _use_k3p && (param.pooling_param.window_h == 2);
    _use_k3p = _use_k3p && (param.pooling_param.window_w == 2);
    _use_k3p = _use_k3p && (param.pooling_param.stride_h == 2);
    _use_k3p = _use_k3p && (param.pooling_param.stride_w == 2);
    _use_k3p = _use_k3p && (param.pooling_param.pad_h == 0);
    _use_k3p = _use_k3p && (param.pooling_param.pad_w == 0);
    _use_k3p = _use_k3p && (param.pooling_param.pooling_type == Pooling_max);

    _use_kp = _use_kp && (param.conv_param.group == 1);
    _use_kp = _use_kp && (param.pooling_param.window_h == 2);
    _use_kp = _use_kp && (param.pooling_param.window_w == 2);
    _use_kp = _use_kp && (param.pooling_param.stride_h == 2);
    _use_kp = _use_kp && (param.pooling_param.stride_w == 2);
    _use_kp = _use_kp && (param.pooling_param.pad_h == 0);
    _use_kp = _use_kp && (param.pooling_param.pad_w == 0);
    _use_kp = _use_kp && (param.pooling_param.pooling_type == Pooling_max);
    _use_kp = _use_kp && (param.conv_param.bias()->valid_size() > 0);

    if (_use_k3p) {
        dispatch_func = winograd_conv_relu_pooling<float, float>;
    } else if (_use_kp) {
        const int K = param.conv_param.weight()->num();

        if (K % 4 == 0) {
            dispatch_func = direct_conv_bias_relu_maxpool2k2s0p_Kdivis4<float, float>;
        } else {
            dispatch_func = direct_conv_bias_relu_maxpool2k2s0p_Kindiv4<float, float>;
        }
    } else {
        _inner_tensor.re_alloc(_inner_shape, AK_FLOAT);

        _inner_tensor_v.resize(1);
        _inner_tensor_v[0] = &_inner_tensor;
        _conv.init(inputs, _inner_tensor_v, param.conv_param, ctx);
        _pool.init(_inner_tensor_v, outputs, param.pooling_param, ctx);
    }

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConv2DPooling<NV, AK_FLOAT>::dispatch(
    const std::vector<Tensor<NV>*>& inputs,
    std::vector<Tensor<NV>*>& outputs,
    ConvPoolingParam<NV>& param) {
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    const float* bias_data = nullptr;
    const float* weight_data = nullptr;

    if (param.conv_param.bias()->size() > 0) {
        bias_data = (const float*)param.conv_param.bias()->data();
    }

    weight_data = (const float*)param.conv_param.weight()->data();

    if (_use_k3p || _use_kp) {
        dispatch_func((const float*)inputs[0]->data(), (float*)outputs[0]->mutable_data(),
                      weight_data,
                      bias_data,
                      inputs[0]->num(),
                      inputs[0]->channel(),
                      inputs[0]->height(),
                      inputs[0]->width(),
                      outputs[0]->channel(),
                      _inner_shape.height(),
                      _inner_shape.width(),
                      shape_in[1],
                      shape_in[2],
                      shape_in[3],
                      shape_out[1],
                      shape_out[2],
                      shape_out[3],
                      _kernel_height,
                      _kernel_width,
                      param.conv_param.pad_h,
                      param.conv_param.pad_w,
                      param.conv_param.stride_h,
                      param.conv_param.stride_w,
                      param.conv_param.dilation_h,
                      param.conv_param.dilation_w,
                      param.conv_param.group,
                      param.conv_param.alpha,
                      param.conv_param.beta,
                      this->_ctx->get_compute_stream());
    } else {
        //                LOG(INFO) << "calling saber_conv + pooling";
        _conv.dispatch(inputs, _inner_tensor_v, param.conv_param);
        _pool.dispatch(_inner_tensor_v, outputs, param.pooling_param);
    }

    return SaberSuccess;
}
template <>
SaberStatus SaberConv2DPooling<NV, AK_FLOAT>::trans_weights(Tensor<NV>& target_weights,
        Tensor<NV>& target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    if (_extern_trans || target_weights.valid_size() == 0) {
        return SaberSuccess;
    }

    bool use_k1s1p0 = true;
    use_k1s1p0 = use_k1s1p0 && (target_weights.height() == 1);
    use_k1s1p0 = use_k1s1p0 && (target_weights.width() == 1);
    use_k1s1p0 = use_k1s1p0 && (pad_h == 0);
    use_k1s1p0 = use_k1s1p0 && (pad_w == 0);
    use_k1s1p0 = use_k1s1p0 && (stride_h == 1);
    use_k1s1p0 = use_k1s1p0 && (stride_w == 1);
    use_k1s1p0 = use_k1s1p0 && (dilation_h == 1);
    use_k1s1p0 = use_k1s1p0 && (dilation_w == 1);
    use_k1s1p0 = use_k1s1p0 && (group == 1);
    use_k1s1p0 = use_k1s1p0 && (target_bias.valid_size() > 0);

    if (use_k1s1p0) {
        _extern_trans = true;
        return SaberSuccess;
    }

    if (_use_vender) {
        _extern_trans = true;
        return SaberSuccess;
    }

    if (target_weights.valid_size() > 0) {
        conv_trans_weights<NV, NVHX86>(target_weights,
                                       stride_h, stride_w, group, true, nullptr, dilation_h, dilation_w);
    }

    _extern_trans = true;
    return SaberSuccess;
}

template <>
SaberStatus SaberConv2DPooling<NV, AK_INT8>::trans_weights(Tensor<NV>& target_weights,
        Tensor<NV>& target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberUnImplError;
}

template <>
SaberStatus SaberConv2DPooling<NV, AK_HALF>::trans_weights(Tensor<NV>& target_weights,
        Tensor<NV>& target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberUnImplError;
}


template class SaberConv2DPooling<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, NV, AK_INT8);
}
}
