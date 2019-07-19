
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/impl/cuda/saber_eltwise.h"
#include "saber/funcs/impl/cuda/saber_conv_eltwise.h"
#include "saber/funcs/impl/cuda/vender_conv.h"
#include "saber/funcs/calibrate.h"
#include "saber_conv_eltwise.h"
#include "sass_funcs.h"
#include "saber/funcs/debug.h"

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
                                       this->_ctx->get_compute_stream(), 1.f, 1.f);
            } else {
                conv_gemm_k1s1p0<false>(num, in_stride, out_stride,
                                        (float*)outputs[0]->mutable_data(),
                                        (const float*)inputs[0]->data(),
                                        (const float*)param.conv_param.weight()->data(),
                                        chout, chin, hin, win, bias_data,
                                        this->_ctx->get_compute_stream(), 1.f, 1.f);
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
    if (_use_k1s1p0) {
        _extern_trans = true;
        return SaberSuccess;
    } else {
        _extern_trans = true;
        return _conv.trans_weights(target_weights, target_bias,
                                   pad_h, pad_w, dilation_h, dilation_w,
                                   stride_h, stride_w, group);
    }
}
template <>
SaberStatus SaberConvEltwise<NV, AK_INT8>::\
        create(const std::vector<Tensor<NV> *>& inputs,
               std::vector<Tensor<NV> *>& outputs,
               ConvEltwiseParam<NV>& param, Context<NV>& ctx) {
    if (_impl != nullptr) {
        return _impl->create(_in_data_tensor, _out_data_tensor, param.conv_param, ctx);
    } else {
        return SaberUnImplError;
    }
}

template <>
SaberStatus SaberConvEltwise<NV, AK_INT8>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvEltwiseParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    // only support 61 arch for now.
    bool arch_check = (generate_arch == 61);

    bool use_k1s1p0 = arch_check;
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.weight()->height() == 1);
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.weight()->width() == 1);
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.pad_h == 0);
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.pad_w == 0);
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.stride_h == 1);
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.stride_w == 1);
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.dilation_h == 1);
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.dilation_w == 1);
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.group == 1);
    use_k1s1p0 = use_k1s1p0 && (param.conv_param.bias()->valid_size() > 0);

    if (!arch_check) {
        LOG(FATAL) << "not support using int8";
    } else {
        if (inputs[0]->get_scale().size() == 1) {
            _in_scale = inputs[0]->get_scale()[0];
        } else {
            LOG(FATAL) << "scale now support static calibrate only!!";
        }
        if (outputs[0]->get_scale().size() == 1) {
            _out_scale = outputs[0]->get_scale()[0];
        } else {
            LOG(FATAL) << "scale now support static calibrate only!!";
        }
    }

    if (arch_check && use_k1s1p0) {
//        LOG(INFO) << " using gemm to run 1x1 conv!!!!";
        _impl = new SaberGemmLikeConv<AK_INT8>;
        _use_vender = false;
    } else {
        LOG(FATAL) << "wrong gpu! This arch is not supporting int8 feature!!";
    }

    if (param.eltwise_param.activation_param.has_active) {
        _impl->set_act(true);
    }

    _in_data_tensor.resize(1);
    _out_data_tensor.resize(1);

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        if (inputs[0]->get_scale().size() == 1) {
            _in_scale = inputs[0]->get_scale()[0];
        } else {
            LOG(FATAL) << "scale now support static calibrate only!!";
        }
        Shape in_shape = inputs[0]->valid_shape();
        int in_channel = in_shape.channel();
        in_shape.set_channel(4 * ((in_channel + 3 ) >> 2));
        int8_input.re_alloc(in_shape, AK_INT8);
        int8_input.set_scale(inputs[0]->get_scale());
        int8_input.set_layout(Layout_NCHW_C4);
        _in_data_tensor[0] = &int8_input;
    } else {
        _in_data_tensor[0] = inputs[0];
    }
    if (outputs[0]->get_dtype() == AK_INT8) {
        _output_int8 = true;
    }
    _out_data_tensor[0] = outputs[0];

    _impl->init(_in_data_tensor, _out_data_tensor, param.conv_param, ctx);
    return create(_in_data_tensor, _out_data_tensor, param, ctx);
}

template <>
SaberStatus SaberConvEltwise<NV, AK_INT8>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ConvEltwiseParam<NV>& param) {

    float in_scale = 0.f;
    if (inputs[0]->get_dtype() == AK_FLOAT) {
        if (inputs[0]->get_scale().size() == 1) {
            in_scale = inputs[0]->get_scale()[0];
        } else {
            LOG(FATAL) << "scale now support static calibrate only!!";
        }
        conv_calibrate_fp32_int8_c4(int8_input, *inputs[0], in_scale, *(this->_ctx));
    }

    if (_impl != nullptr) {
        _impl->dispatch(_in_data_tensor, _out_data_tensor, param.conv_param);
    }
    const float* weights_scale = (const float*)param.conv_param.weight()->get_scale_data();
    if (param.conv_param.weight()->get_scale().size() > 1) {
        conv_calibrate_int32_fp32(
                *_out_data_tensor[0], *_out_data_tensor[0], in_scale, weights_scale, *_ctx);
    }

    return SaberSuccess;
}
template <>
SaberStatus SaberConvEltwise<NV, AK_INT8>::trans_weights(
        Tensor<NV> &target_weights, Tensor<NV> &target_bias,
        int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    // only support 61 arch for now.
    bool arch_check = (generate_arch == 61);

    if (target_weights.valid_size() == 0) {
        return SaberSuccess;
    }
    if (target_weights.get_dtype() == AK_INT8) {
        return SaberSuccess;
    }
    bool use_k1s1p0 = arch_check;
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

    if (arch_check && !use_k1s1p0 && !_use_vender) {
        Tensor<NVHX86> weights_host;
        Tensor<NVHX86> weights_temp;
        weights_host.re_alloc(target_weights.valid_shape(), AK_FLOAT);
        Shape target_weights_shape = target_weights.valid_shape();
        int in_channel = target_weights_shape.channel();
        target_weights_shape.set_channel(4 * ((in_channel + 3 ) >> 2));
        weights_temp.re_alloc(target_weights_shape, AK_INT8);
        weights_host.copy_from(target_weights);

        convert_weights_to_direct(weights_temp, weights_host, *_ctx, _scale_per_k);
        target_weights.set_dtype(AK_INT8);
        target_weights.re_alloc(target_weights_shape, AK_INT8);
        target_weights.set_layout(Layout_NCHW_C4);
        target_weights.copy_from(weights_temp);
        target_weights.set_scale(weights_temp.get_scale());
        if (_output_int8 && target_bias.valid_size() > 0) {
//            LOG(INFO) << "scale bias with out_scale: " << _out_scale;
            Tensor<NVHX86> bias_fp32_host;
            Tensor<NVHX86> bias_int32_host;
            bias_fp32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_int32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_fp32_host.copy_from(target_bias);
            convert_bias_host(bias_int32_host, bias_fp32_host, _out_scale,
                              {1.f}, *_ctx, _scale_per_k);
            target_bias.copy_from(bias_int32_host);
        }
        return SaberSuccess;

    } else if (arch_check
               && target_weights.num() % 4 == 0) {

        // prepare int8 memory
        Tensor<NVHX86> weights_fp32_host;
        Tensor<NVHX86> weights_int8_host;
        weights_fp32_host.re_alloc(target_weights.valid_shape(), AK_FLOAT);
        Shape target_weights_shape = target_weights.valid_shape();
        int in_channel = target_weights_shape.channel();
        target_weights_shape.set_channel(4 * ((in_channel + 3 ) >> 2));
        weights_int8_host.re_alloc(target_weights_shape, AK_INT8);
        weights_int8_host.set_layout(Layout_NCHW_C4);
        weights_fp32_host.copy_from(target_weights);
        convert_weights_to_nchw_c4_host(weights_int8_host, weights_fp32_host, *_ctx, _scale_per_k);
        // Open this will be an inplace trans

        target_weights.set_dtype(AK_INT8);
        target_weights.re_alloc(target_weights_shape, AK_INT8);
        target_weights.set_layout(Layout_NCHW_C4);
        target_weights.copy_from(weights_int8_host);
        target_weights.set_scale(weights_int8_host.get_scale());
        if (_use_vender && target_bias.valid_size() > 0) {
            Tensor<NVHX86> bias_fp32_host;
            Tensor<NVHX86> bias_int32_host;
            bias_fp32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_int32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_fp32_host.copy_from(target_bias);
            if (_output_int8) {
                convert_bias_host(bias_int32_host, bias_fp32_host, _in_scale,
                                  target_weights.get_scale(), *_ctx, _scale_per_k);
            } else {
                convert_bias_host(bias_int32_host, bias_fp32_host, _in_scale / _out_scale,
                                  target_weights.get_scale(), *_ctx, _scale_per_k);
            }
            target_bias.copy_from(bias_int32_host);
        } else if (_output_int8 && target_bias.valid_size() > 0) {
//            LOG(INFO) << "scale bias with out_scale: " << _out_scale;
            Tensor<NVHX86> bias_fp32_host;
            Tensor<NVHX86> bias_int32_host;
            bias_fp32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_int32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_fp32_host.copy_from(target_bias);
//            LOG(INFO) << _out_scale;
            convert_bias_host(bias_int32_host, bias_fp32_host, _out_scale,
                              {1.f}, *_ctx, _scale_per_k);
            target_bias.copy_from(bias_int32_host);
        }
    } else {
        LOG(FATAL) << "gpu arch error!!!";
    }
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
template class SaberConvEltwise<NV, AK_INT8>;
DEFINE_OP_TEMPLATE(SaberConvEltwise, ConvEltwiseParam, NV, AK_HALF);
}
}
