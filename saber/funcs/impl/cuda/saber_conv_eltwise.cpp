
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/impl/cuda/saber_eltwise.h"
#include "saber/funcs/impl/cuda/saber_conv_eltwise.h"
#include "sass_funcs.h"
#include "saber/funcs/calibrate.h"
#include "saber_conv_eltwise.h"
#include "saber/funcs/impl/cuda/vender_conv.h"
namespace anakin {
namespace saber {

template <>
SaberStatus SaberConvEltwise<NV, AK_FLOAT>::\
        create(const std::vector<Tensor<NV> *>& inputs,
               std::vector<Tensor<NV> *>& outputs,
               ConvEltwiseParam<NV>& param, Context<NV>& ctx) {

    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    if (!_use_k1s1p0 && !_use_k3) {
        _inner_tensor.reshape(_inner_shape);
        _inner_tensor_v.resize(2);
        _inner_tensor_v[0] = &_inner_tensor;
        _conv.create(inputs, _inner_tensor_v, param.conv_param, ctx);
        _eltwise.create(_inner_tensor_v, outputs, param.eltwise_param, ctx);
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberConvEltwise<NV, AK_FLOAT>::
init(const std::vector<Tensor<NV> *>& inputs,
     std::vector<Tensor<NV> *>& outputs,
     ConvEltwiseParam<NV>& param, Context<NV>& ctx) {

    _ctx = &ctx;
    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    bool arch_check = (generate_arch == 50) || (generate_arch == 61);

    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _kernel_height = param.conv_param.weight()->height();
    _kernel_width = param.conv_param.weight()->width();

    _use_k1s1p0 = true;
    _use_k3 = false; // disable 3x3 kernel

    _use_k1s1p0 = _use_k1s1p0 && arch_check;
    _use_k1s1p0 = _use_k1s1p0 && (_kernel_height == 1);
    _use_k1s1p0 = _use_k1s1p0 && (_kernel_width == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.pad_h == 0);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.pad_w == 0);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.stride_h == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.stride_w == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.dilation_h == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.dilation_w == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.bias()->valid_size() > 0);
    _use_k1s1p0 = _use_k1s1p0 && (!param.conv_param.activation_param.has_active);
    _use_k1s1p0 = _use_k1s1p0 && (param.eltwise_param.operation == Eltwise_sum);
    _use_k1s1p0 = _use_k1s1p0 && (!param.conv_param.activation_param.has_active);

    _use_k3 = _use_k3 && arch_check;
    _use_k3 = _use_k3 && (param.conv_param.weight()->height() == 3);
    _use_k3 = _use_k3 && (param.conv_param.weight()->width() == 3);
    _use_k3 = _use_k3 && (param.conv_param.stride_h == 1);
    _use_k3 = _use_k3 && (param.conv_param.stride_w == 1);
    _use_k3 = _use_k3 && (param.conv_param.dilation_h == 1);
    _use_k3 = _use_k3 && (param.conv_param.dilation_w == 1);
    _use_k3 = _use_k3 && (param.conv_param.group == 1);
    _use_k3 = _use_k3 && (param.eltwise_param.operation == Eltwise_sum);

    if (_use_k1s1p0) {
        return SaberSuccess;
    } else if (_use_k3) {
        dispatch_func_elt = winograd_conv_eltwise<float, float>;
    } else {
        _inner_tensor.re_alloc(_inner_shape, AK_FLOAT);
        _inner_tensor_v.resize(2);
        _inner_tensor_v[0] = &_inner_tensor;
        _conv.init(inputs, _inner_tensor_v, param.conv_param, ctx);
        _eltwise.init(_inner_tensor_v, outputs, param.eltwise_param, ctx);
    }
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConvEltwise<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ConvEltwiseParam<NV>& param) {

    const float* bias_data;
    if (param.conv_param.bias()->size() > 0) {
        bias_data = (const float*)param.conv_param.bias()->data();
    } else {
        bias_data = nullptr;
    }
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chout = outputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int in_stride = chin * win * hin;
    int out_stride = chout * wout * hout;
    if (_use_k1s1p0) {
        if (param.eltwise_param.has_eltwise) {
            if (param.eltwise_param.activation_param.has_active) {
                conv_gemm_k1s1p0<true>(num, in_stride, out_stride,
                                       (float*)outputs[0]->mutable_data(),
                                       (const float*)inputs[0]->data(),
                                       (const float*)param.conv_param.weight()->data(),
                                       chout, chin, hin, win, bias_data,
                                       this->_ctx->get_compute_stream(),1.f, 1.f);
            } else {
                conv_gemm_k1s1p0<false>(num, in_stride, out_stride,
                                        (float*)outputs[0]->mutable_data(),
                                        (const float*)inputs[0]->data(),
                                        (const float*)param.conv_param.weight()->data(),
                                        chout, chin, hin, win, bias_data,
                                        this->_ctx->get_compute_stream(),1.f, 1.f);
            }
        } else {
            if (param.conv_param.activation_param.has_active) {
                conv_gemm_k1s1p0<true>(num, in_stride, out_stride,
                                       (float*)outputs[0]->mutable_data(),
                                       (const float*)inputs[0]->data(),
                                       (const float*)param.conv_param.weight()->data(),
                                       chout, chin, hin, win, bias_data,
                                       this->_ctx->get_compute_stream(), 1.f, 0.f);
            } else {
                conv_gemm_k1s1p0<false>(num, in_stride, out_stride,
                                        (float*)outputs[0]->mutable_data(),
                                        (const float*)inputs[0]->data(),
                                        (const float*)param.conv_param.weight()->data(),
                                        chout, chin, hin, win, bias_data,
                                        this->_ctx->get_compute_stream(), 1.f, 0.f);
            }
        }
        return SaberSuccess;
    } else if (_use_k3){
        dispatch_func_elt((const float*)inputs[0]->data(),
                          (float*)outputs[0]->mutable_data(),
                          (const float*)param.conv_param.weight()->data(),
                          bias_data, num, chin, hin, win,
                          chout, hout, wout,
                          shape_in[1],
                          shape_in[2],
                          shape_in[3],
                          shape_out[1],
                          shape_out[2],
                          shape_out[3],
                          _kernel_height, _kernel_width,
                          param.conv_param.pad_h,
                          param.conv_param.pad_w,
                          param.conv_param.stride_h,
                          param.conv_param.stride_w,
                          param.conv_param.dilation_h,
                          param.conv_param.dilation_w,
                          param.conv_param.group,
                          param.conv_param.alpha,
                          param.conv_param.beta,
                          param.eltwise_param.operation,
                          this->_ctx->get_compute_stream());
        return SaberSuccess;
    } else {
        _conv.dispatch(inputs, _inner_tensor_v, param.conv_param);
        _inner_tensor_v[1] = outputs[0];
        _eltwise.dispatch(_inner_tensor_v, outputs, param.eltwise_param);
    }
    return SaberSuccess;
}
template <>
SaberStatus SaberConvEltwise<NV, AK_FLOAT>::trans_weights(
        Tensor<NV> &target_weights, Tensor<NV> &target_bias,
        int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    bool arch_check = (generate_arch == 50) || (generate_arch == 61);
    if (!arch_check) {
        return SaberSuccess;
    }

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
SaberStatus SaberConvEltwise<NV, AK_INT8>::trans_weights(
        Tensor<NV> &target_weights, Tensor<NV> &target_bias,
        int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberSuccess;
}
template <>
SaberStatus SaberConvEltwise<NV, AK_HALF>::trans_weights(
        Tensor<NV> &target_weights, Tensor<NV> &target_bias,
        int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberSuccess;
}

template class SaberConvEltwise<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConvEltwise, ConvEltwiseParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConvEltwise, ConvEltwiseParam, NV, AK_INT8);
}
}
