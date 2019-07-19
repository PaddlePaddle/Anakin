
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/calibrate.h"
#include "saber/funcs/timer.h"
#include "saber/funcs/impl/cuda/saber_conv_depthwise.h"
#include "saber/funcs/impl/cuda/saber_conv_direct.h"
#include "saber/funcs/impl/cuda/saber_conv_gemmlike.h"
#include "saber/funcs/impl/cuda/saber_conv_winograd.h"
#include "saber/funcs/impl/cuda/vender_conv.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/debug.h"

namespace anakin {
namespace saber {

template <>
void SaberConv2D<NV, AK_FLOAT>::find_fastest_alg(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {

    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    bool arch_check = (generate_arch == 50) || (generate_arch == 61);

    bool use_k1s1p0 = arch_check;
    bool use_k3s1 = arch_check;
    bool use_direct = arch_check;
    bool use_depthwise = true;

    use_k1s1p0 = use_k1s1p0 && (param.weight()->height() == 1);
    use_k1s1p0 = use_k1s1p0 && (param.weight()->width() == 1);
    use_k1s1p0 = use_k1s1p0 && (param.pad_h == 0);
    use_k1s1p0 = use_k1s1p0 && (param.pad_w == 0);
    use_k1s1p0 = use_k1s1p0 && (param.stride_h == 1);
    use_k1s1p0 = use_k1s1p0 && (param.stride_w == 1);
    use_k1s1p0 = use_k1s1p0 && (param.dilation_h == 1);
    use_k1s1p0 = use_k1s1p0 && (param.dilation_w == 1);
    use_k1s1p0 = use_k1s1p0 && (param.group == 1);
    use_k1s1p0 = use_k1s1p0 && (param.bias()->valid_size() > 0);

    use_k3s1 = use_k3s1 && (param.stride_h == 1);
    use_k3s1 = use_k3s1 && (param.stride_w == 1);
    use_k3s1 = use_k3s1 && (param.weight()->height() == 3);
    use_k3s1 = use_k3s1 && (param.weight()->width() == 3);
    use_k3s1 = use_k3s1 && (param.dilation_h == 1);
    use_k3s1 = use_k3s1 && (param.dilation_w == 1);
    use_k3s1 = use_k3s1 && (param.group == 1);

    use_direct = use_direct && (param.group == 1);
    use_direct = use_direct && (inputs[0]->height() > 8);
    use_direct = use_direct && (inputs[0]->width() > 8);

    use_depthwise = use_depthwise && (param.group == inputs[0]->channel());
    use_depthwise = use_depthwise && (param.group == outputs[0]->channel());

    if (use_k1s1p0) {
        _kernel_alg = K_k1s1p0;
    } else if (use_k3s1) {
        _kernel_alg = K_k3s1;
    } else if (use_direct) {
        _kernel_alg = K_direct;
    } else if (use_depthwise) {
        _kernel_alg = K_depthwise;
    } else {
        _kernel_alg = K_vender;
    }
}

template <>
SaberStatus SaberConv2D<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {
    KernelAlg last_alg = _kernel_alg;
    find_fastest_alg(inputs, outputs, param, ctx);
    if (_kernel_alg != last_alg) {
        // bad case.
        if (_impl != nullptr) {
            delete _impl;
        }
        if (_kernel_alg == K_direct) {
//            LOG(INFO) << "change to use direct!!!";
            _impl = new SaberDirectConv<AK_FLOAT>;
            return _impl->init(inputs, outputs, param, ctx);
        } else if (_kernel_alg == K_vender) {
//            LOG(INFO) << "change to use vender!!!!";
            _impl = new VenderConv2D<NV, AK_FLOAT>;
            dynamic_cast<VenderConv2D<NV, AK_FLOAT> *>(
                    this->_impl)->load_origin_weight(_origin_weight, ctx);
            return _impl->init(inputs, outputs, param, ctx);
        } else {
            LOG(FATAL) << "this situation should not happened!!";
        }

    }
    if (_impl != nullptr) {
        return _impl->create(inputs, outputs, param, ctx);
    } else {
        return SaberUnImplError;
    }
}

template <>
SaberStatus SaberConv2D<NV, AK_FLOAT>::init(const std::vector<Tensor<NV> *>& inputs,
                 std::vector<Tensor<NV> *>& outputs,
                 ConvParam<NV>& param, Context<NV> &ctx) {
    this->_ctx = &ctx;
//    LOG(INFO) << "only copy once!!!";
    _origin_weight.re_alloc(param.weight()->valid_shape(), param.weight()->get_dtype());
    _origin_weight.async_copy_from(*param.weight(), ctx.get_compute_stream());
    if (_impl == nullptr) {
        find_fastest_alg(inputs, outputs, param, ctx);

        if (_kernel_alg == K_k1s1p0) {
            _impl = new SaberGemmLikeConv<AK_FLOAT>;
        } else if (_kernel_alg == K_k3s1) {

            this->_impl = new SaberWinogradConv<AK_FLOAT>;
        } else if (_kernel_alg == K_direct) {

            _impl = new SaberDirectConv<AK_FLOAT>;
        } else if (_kernel_alg == K_depthwise) {

            _impl = new SaberDepthWiseConv<AK_FLOAT>;
        } else {
            // I will never fail!!!
            _use_vender = true;
            _impl = new VenderConv2D<NV, AK_FLOAT>;
        }
    }
    this->_impl->init(inputs, outputs, param, ctx);
    return create(inputs, outputs, param, ctx);
}
template <>
SaberStatus SaberConv2D<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param) {
    if (_impl != nullptr) {
        return _impl->dispatch(inputs, outputs, param);
    } else {
        return SaberUnImplError;
    }
}

template <>
SaberStatus SaberConv2D<NV, AK_FLOAT>::trans_weights(Tensor<NV> &target_weights,
        Tensor<NV> &target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
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
    if (group == target_weights.channel() && group == target_weights.num()) {
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
SaberStatus SaberConv2D<NV, AK_INT8>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        if (inputs[0]->get_scale().size() == 1) {
            _in_scale = inputs[0]->get_scale()[0];
        } else {
            LOG(FATAL) << "scale now support static calibrate only!!";
        }
        Shape in_shape = inputs[0]->valid_shape();
        int in_channel = in_shape.channel();
        in_shape.set_channel(4 * ((in_channel + 3) >> 2));
        int8_input.re_alloc(in_shape, AK_INT8);
        int8_input.set_scale(inputs[0]->get_scale());
        int8_input.set_layout(Layout_NCHW_C4);
        _in_data_tensor[0] = &int8_input;
    } else {
        _in_data_tensor[0] = inputs[0];
    }
    if (outputs[0]->get_dtype() == AK_FLOAT) {
        _out_data_tensor[0] = outputs[0];
    } else if (outputs[0]->get_dtype() == AK_INT8) {
        if (!_scale_per_k) {
            _output_int8 = true;
            _out_data_tensor[0] = outputs[0];
        } else {
            Shape out_shape = outputs[0]->valid_shape();
            int out_channel = out_shape.channel();
            out_shape.set_channel(4 * ((out_channel + 3) >> 2));
            int8_output.re_alloc(out_shape, AK_FLOAT);
            int8_output.set_layout(Layout_NCHW);
            int8_output.set_scale(outputs[0]->get_scale());
            _out_data_tensor[0] = &int8_output;
        }
    } else {
        LOG(FATAL) << " out dtype error!!";
    }
    if (_impl != nullptr) {
        return _impl->create(_in_data_tensor, _out_data_tensor, param, ctx);
    } else {
        return SaberUnImplError;
    }
}

template <>
SaberStatus SaberConv2D<NV, AK_INT8>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {
    this->_ctx = &ctx;
    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    // only support 61 arch for now.
    bool arch_check = (generate_arch == 61);

    bool use_k1s1p0 = arch_check;
    use_k1s1p0 = use_k1s1p0 && (param.weight()->height() == 1);
    use_k1s1p0 = use_k1s1p0 && (param.weight()->width() == 1);
    use_k1s1p0 = use_k1s1p0 && (param.pad_h == 0);
    use_k1s1p0 = use_k1s1p0 && (param.pad_w == 0);
    use_k1s1p0 = use_k1s1p0 && (param.stride_h == 1);
    use_k1s1p0 = use_k1s1p0 && (param.stride_w == 1);
    use_k1s1p0 = use_k1s1p0 && (param.dilation_h == 1);
    use_k1s1p0 = use_k1s1p0 && (param.dilation_w == 1);
    use_k1s1p0 = use_k1s1p0 && (param.group == 1);
    use_k1s1p0 = use_k1s1p0 && (param.bias()->valid_size() > 0);

    bool use_depthwise = true;
    use_depthwise = use_depthwise && (param.group == inputs[0]->channel());
    use_depthwise = use_depthwise && (param.group == outputs[0]->channel());
    if (param.weight()->get_scale().size() > 1) {
        _scale_per_k = true;
    }
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
    if (use_depthwise) {
        _impl = new SaberDepthWiseConv<AK_INT8>;
        _use_vender = false;
    } else if (arch_check && use_k1s1p0) {
//        LOG(INFO) << " using gemm to run 1x1 conv!!!!";
        _impl = new SaberGemmLikeConv<AK_INT8>;
        _use_vender = false;
    } else if (arch_check
           && (outputs[0]->channel() % 4) == 0
           && (inputs[0]->height() <= 128)
           && (inputs[0]->width() <= 128)) {
//        LOG(INFO) << " direct alg seleted!!!!";
        _impl = new SaberDirectConv<AK_INT8>;
        _use_vender = false;
    } else if (arch_check) {
        _impl = new VenderConv2D<NV, AK_INT8>;
        _use_vender = true;
    } else {
        LOG(FATAL) << "wrong gpu! This arch is not supporting int8 feature!!";
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
        in_shape.set_channel(4 * ((in_channel + 3) >> 2));
        int8_input.re_alloc(in_shape, AK_INT8);
        int8_input.set_scale(inputs[0]->get_scale());
        int8_input.set_layout(Layout_NCHW_C4);
        _in_data_tensor[0] = &int8_input;
    } else {
        _in_data_tensor[0] = inputs[0];
    }
    if (outputs[0]->get_dtype() == AK_FLOAT) {
        _out_data_tensor[0] = outputs[0];
    } else if (outputs[0]->get_dtype() == AK_INT8) {
        if (!_scale_per_k) {
            _output_int8 = true;
            _out_data_tensor[0] = outputs[0];
        } else {
            Shape out_shape = outputs[0]->valid_shape();
            int out_channel = out_shape.channel();
            out_shape.set_channel(4 * ((out_channel + 3) >> 2));
            int8_output.re_alloc(out_shape, AK_FLOAT);
            int8_output.set_layout(Layout_NCHW);
            int8_output.set_scale(outputs[0]->get_scale());
            _out_data_tensor[0] = &int8_output;
        }
    } else {
        LOG(FATAL) << " out dtype error!!";
    }
    _impl->init(_in_data_tensor, _out_data_tensor, param, ctx);
    return create(_in_data_tensor, _out_data_tensor, param, ctx);
}

template <>
SaberStatus SaberConv2D<NV, AK_INT8>::dispatch(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param) {

    float in_scale = 0.f;
    // get input scale for scale weights per k
    if (inputs[0]->get_scale().size() == 1) {
        in_scale = inputs[0]->get_scale()[0];
    } else {
        LOG(FATAL) << "scale now support static calibrate only!!";
    }
    if (inputs[0]->get_dtype() == AK_FLOAT) {
        conv_calibrate_fp32_int8_c4(int8_input, *inputs[0], in_scale, *(this->_ctx));
    }

    if (_impl != nullptr) {
        _impl->dispatch(_in_data_tensor, _out_data_tensor, param);
    }
    const float* weights_scale = (const float*)param.weight()->get_scale_data();

    if (_out_data_tensor[0]->get_dtype() == AK_FLOAT
        && param.weight()->get_scale().size() > 1) {
        conv_calibrate_int32_fp32(
                *_out_data_tensor[0], *_out_data_tensor[0], in_scale, weights_scale, *_ctx);
    }
    // with scale_per_k if output is directly support scale remove me
    if (outputs[0]->get_dtype() == AK_INT8 && _scale_per_k) {
        float out_scale = 0.f;
        if (outputs[0]->get_scale().size() == 1) {
            out_scale = outputs[0]->get_scale()[0];
        } else {
            LOG(FATAL) << "scale now support static calibrate only!!";
        }
        conv_calibrate_fp32_int8_c4(*outputs[0], *_out_data_tensor[0],
                out_scale, *(this->_ctx));
    }

    return SaberSuccess;
}
template <>
SaberStatus SaberConv2D<NV, AK_INT8>::trans_weights(
        Tensor<NV> &target_weights,
        Tensor<NV> &target_bias,
        int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {

    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    // only support 61 arch for now.
    bool arch_check = (generate_arch == 61);

    if (target_weights.valid_size() == 0) {
        return SaberSuccess;
    }
    if (target_weights.get_scale().size() > 1) {
        _scale_per_k = true;
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

    bool use_depthwise = true;
    use_depthwise = use_depthwise && (group == target_weights.num());
    if (use_depthwise) {
        if (target_weights.get_dtype() == AK_FLOAT) {
            Tensor<NVHX86> weights_fp32_host;
            Tensor<NVHX86> weights_int8_host;
            weights_fp32_host.re_alloc(target_weights.valid_shape(), AK_FLOAT);
            Shape target_weights_shape = target_weights.valid_shape();
            int num = target_weights_shape.num();
            target_weights_shape.set_num(4 * ((num + 3) >> 2));
            weights_int8_host.re_alloc(target_weights_shape, AK_INT8);
//        weights_int8_host.set_layout(Layout_NCHW_C4);
            weights_fp32_host.copy_from(target_weights);
            convert_weights_to_depthwise(weights_int8_host, weights_fp32_host, *_ctx, _scale_per_k);
            target_weights.set_dtype(AK_INT8);
            target_weights.re_alloc(target_weights_shape, AK_INT8);
//        target_weights.set_layout(Layout_NCHW_C4);
            target_weights.copy_from(weights_int8_host);
            target_weights.set_scale(weights_int8_host.get_scale());
        } else {
            Tensor<NVHX86> weights_int8_host;
            Tensor<NVHX86> weights_int8_trans_host;
            Shape target_weights_shape = target_weights.valid_shape();
            int num = target_weights_shape.num();
            target_weights_shape.set_num(4 * ((num + 3) >> 2));
            weights_int8_host.re_alloc(target_weights.valid_shape(), AK_INT8);
            weights_int8_host.copy_from(target_weights);
            weights_int8_trans_host.re_alloc(target_weights_shape, AK_INT8);
            target_weights.re_alloc(target_weights_shape, AK_INT8);
            int height = weights_int8_host.height();
            int width = weights_int8_host.width();
            layout_trans_depthwise<char>(
                    (char*)weights_int8_trans_host.mutable_data(),
                    (const char*)weights_int8_host.data(),
                    num, height, width);
            target_weights.copy_from(weights_int8_trans_host);
            target_weights_shape.set_num(num);
            target_weights.set_shape(target_weights_shape);
        }
        if (_output_int8 && target_bias.valid_size() > 0) {
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
        return SaberSuccess;
    }
    if (arch_check && !use_k1s1p0 && !_use_vender) {
        if (target_weights.get_dtype() == AK_FLOAT) {
            Tensor<NVHX86> weights_host;
            Tensor<NVHX86> weights_temp;
            weights_host.re_alloc(target_weights.valid_shape(), AK_FLOAT);
            Shape target_weights_shape = target_weights.valid_shape();
            int in_channel = target_weights_shape.channel();
            target_weights_shape.set_channel(4 * ((in_channel + 3) >> 2));
            weights_temp.re_alloc(target_weights_shape, AK_INT8);
            weights_host.copy_from(target_weights);

            convert_weights_to_direct(weights_temp, weights_host, *_ctx, _scale_per_k);
            target_weights.set_dtype(AK_INT8);
            target_weights.re_alloc(target_weights_shape, AK_INT8);
            target_weights.set_layout(Layout_NCHW_C4);
            target_weights.copy_from(weights_temp);
            target_weights.set_scale(weights_temp.get_scale());
        } else {
            Tensor<NVHX86> weight_int8_host;
            Tensor<NVHX86> weight_temp;
            Tensor<NVHX86> weight_temp2;
            weight_int8_host.re_alloc(target_weights.valid_shape(), AK_INT8);
            weight_temp.re_alloc(target_weights.valid_shape(), AK_INT8);
            weight_temp2.re_alloc(target_weights.valid_shape(), AK_INT8);
            weight_int8_host.copy_from(target_weights);
            transpose_filter_kcrs_2_crskc4(
                    (const char *) weight_int8_host.data(),
                    (char *) weight_temp.mutable_data(),
                    (char *) weight_temp2.mutable_data(),
                    weight_int8_host.num(),
                    weight_int8_host.channel(),
                    weight_int8_host.height(),
                    weight_int8_host.width());
            target_weights.copy_from(weight_temp2);
        }
        if (_output_int8 && target_bias.valid_size() > 0) {
//            LOG(INFO) << "scale bias with out_scale: " << _out_scale;
            Tensor<NVHX86> bias_fp32_host;
            Tensor<NVHX86> bias_int32_host;
            bias_fp32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_int32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_fp32_host.copy_from(target_bias);
            convert_bias_host(bias_int32_host, bias_fp32_host, _out_scale, {1.f}, *_ctx, _scale_per_k);
            target_bias.copy_from(bias_int32_host);
        }
        return SaberSuccess;

    } else if (arch_check) {

        if (target_weights.get_dtype() == AK_FLOAT) {
            // prepare int8 memory
            Tensor<NVHX86> weights_fp32_host;
            Tensor<NVHX86> weights_int8_host;
            weights_fp32_host.re_alloc(target_weights.valid_shape(), AK_FLOAT);
            Shape target_weights_shape = target_weights.valid_shape();
            int in_channel = target_weights_shape.channel();
            target_weights_shape.set_channel(4 * ((in_channel + 3) >> 2));
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
        } else {
            Tensor<NVHX86> weight_int8_host;
            Tensor<NVHX86> weight_temp;
            weight_int8_host.re_alloc(target_weights.valid_shape(), AK_INT8);
            weight_int8_host.copy_from(target_weights);

            // gives a larger weight shape
            Shape target_weights_shape = target_weights.valid_shape();
            int in_channel = target_weights_shape.channel();
            target_weights_shape.set_layout(Layout_NCHW_C4);
            target_weights_shape.set_channel(4 * ((in_channel + 3) >> 2));

            weight_temp.re_alloc(target_weights_shape, AK_INT8);
            fill_tensor_const(weight_temp, 0);
            target_weights.re_alloc(target_weights_shape, AK_INT8);

            transpose_weight_nchw_2_nchwc4((const char *) weight_int8_host.data(),
                                        (char *) weight_temp.mutable_data(),
                                        weight_int8_host.num(),
                                        weight_int8_host.channel(),
                                        weight_int8_host.height(),
                                        weight_int8_host.width());

            target_weights.copy_from(weight_temp);
        }
        if (_output_int8 && target_bias.valid_size() > 0) {
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
        if (_use_vender && (!_output_int8)
            && _scale_per_k && target_bias.valid_size() > 0) {

            Tensor<NVHX86> bias_fp32_host;
            Tensor<NVHX86> bias_int32_host;
            bias_fp32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_int32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_fp32_host.copy_from(target_bias);
            convert_bias_host(bias_int32_host, bias_fp32_host, _in_scale,
                              target_weights.get_scale(), *_ctx, _scale_per_k);
            target_bias.copy_from(bias_int32_host);
        }
    } else {
        LOG(FATAL) << "gpu arch error!!!";
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberConv2D<NV, AK_HALF>::trans_weights(Tensor<NV> &target_weights,
        Tensor<NV> &target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {

    return SaberUnImplError;

}
DEFINE_OP_TEMPLATE(SaberConv2D, ConvParam, NV, AK_HALF);
}
}
