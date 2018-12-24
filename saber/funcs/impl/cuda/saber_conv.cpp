
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/calibrate.h"
#include "saber/funcs/timer.h"
#include "saber/funcs/impl/cuda/saber_conv_depthwise.h"
#include "saber/funcs/impl/cuda/saber_conv_direct.h"
#include "saber/funcs/impl/cuda/saber_conv_gemmlike.h"
#include "saber/funcs/impl/cuda/saber_conv_winograd.h"
#include "saber/funcs/impl/cuda/vender_conv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberConv2D<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {
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
    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    bool arch_check = (generate_arch == 50) || (generate_arch == 61);
    if (_impl == nullptr) {
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
        if (arch_check && use_k1s1p0) {
            _impl = new SaberGemmLikeConv<AK_FLOAT>;
        } else if (arch_check && param.stride_h == 1 &&
                   param.stride_w == 1 &&
                   param.weight()->height() == 3 &&
                   param.weight()->width() == 3 &&
                   param.dilation_h == 1 &&
                   param.dilation_w == 1 &&
                   param.group == 1) {

            this->_impl = new SaberWinogradConv<AK_FLOAT>;
        } else if (arch_check && param.group == 1) {
            //TODO [zs] This will be a good feature to check if the kernel is out performance of cudnn!!!!
            //TODO this will remove the bad case of saber
            //TODO Better to extract this as a function, whose template is a specify Conv, return(bool) if faster than cudnn
//            SaberDirectConv<AK_FLOAT> temp;
//            VenderConv2D<NV, AK_FLOAT> vender_temp;
//            temp.init(inputs, outputs, param, ctx);
//            vender_temp.init(inputs, outputs, param, ctx);
//            SaberTimer<NV> s_t, v_t;
//            temp.dispatch(inputs, outputs, param);
//            s_t.start(ctx);
//            for (int i = 0; i < 10; ++i) {
//                temp.dispatch(inputs, outputs, param);
//            }
//            s_t.end(ctx);
//            v_t.start(ctx);
//            for (int i = 0; i < 10; ++i) {
//                vender_temp.dispatch(inputs, outputs, param);
//            }
//            v_t.end(ctx);
//            if (v_t.get_average_ms() < s_t.get_average_ms()) {
//                _use_vender = true;
//                this->_impl = new VenderConv2D<NV, AK_FLOAT>;
//            } else {
//                _impl = new SaberDirectConv<AK_FLOAT>;
//            }
            _impl = new SaberDirectConv<AK_FLOAT>;
        } else if (param.group == inputs[0]->channel() && param.group == outputs[0]->channel()) {
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
        int8_input.re_alloc(inputs[0]->valid_shape(), AK_INT8);
        int8_input.set_scale(inputs[0]->get_scale());
        int8_input.set_layout(Layout_NCHW_C4);
    }

    if (outputs[0]->get_dtype() == AK_INT8) {
        if (outputs[0]->get_layout() != Layout_NCHW_C4) {
                    LOG(ERROR) << "output layout must be NCHW_C4 for nv gpu";
        }
        int8_output.re_alloc(outputs[0]->valid_shape(), AK_FLOAT);
        int8_output.set_scale(inputs[0]->get_scale());
        int8_output.set_layout(Layout_NCHW);
    }
    if (_impl != nullptr) {
        return _impl->create(inputs, outputs, param, ctx);
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

    bool use_int8 = true;
    use_int8 &= ((inputs[0]->channel() % 4) == 0);
    use_int8 &= ((outputs[0]->channel() % 4) == 0);
    // INT8 only support Active relu
    use_int8 &= ((!param.activation_param.has_active)
                 || (param.activation_param.active == Active_relu));

    if (!use_int8) {
        return SaberInvalidValue;
    } else {
        if (inputs[0]->get_scale().size() == 1) {
            _in_scale = inputs[0]->get_scale()[0];
        } else {
            LOG(FATAL) << "scale now support static calibrate only!!";
        }
    }
    if (arch_check && use_k1s1p0) {
//        LOG(INFO) << " using gemm to run 1x1 conv!!!!";
        _impl = new SaberGemmLikeConv<AK_INT8>;
        _use_vender = false;
    } else if (arch_check && (outputs[0]->channel() % 4) == 0) {
//        LOG(INFO) << " direct alg seleted!!!!";
        _impl = new SaberDirectConv<AK_INT8>;
        _use_vender = false;
    } else if (arch_check) {
        _impl = new VenderConv2D<NV, AK_INT8>;
        _use_vender = true;
    } else {
        LOG(FATAL) << "wrong gpu! This arch is not supporting int8 feature!!";
    }
    _impl->init(inputs, outputs, param, ctx);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConv2D<NV, AK_INT8>::dispatch(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param) {
    std::vector<Tensor<NV> *> in_data_tensor;
    std::vector<Tensor<NV> *> out_data_tensor;

    in_data_tensor.resize(1);
    out_data_tensor.resize(1);
    float in_scale = 0.f;

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        if (inputs[0]->get_scale().size() == 1) {
            in_scale = inputs[0]->get_scale()[0];
        } else {
            LOG(FATAL) << "scale now support static calibrate only!!";
        }
        conv_calibrate_fp32_int8_c4(int8_input, *inputs[0], in_scale, *(this->_ctx));
        in_data_tensor[0] = &int8_input;
    } else {
        in_data_tensor[0] = inputs[0];
    }
    if (outputs[0]->get_dtype() == AK_INT8) {
        if (outputs[0]->get_layout() != Layout_NCHW_C4) {
            LOG(ERROR) << "output layout must be NCHW_C4 for nv gpu";
        }
        out_data_tensor[0] = &int8_output;
    } else {
        out_data_tensor[0] = outputs[0];
    }

    if (_impl != nullptr) {
        _impl->dispatch(in_data_tensor, out_data_tensor, param);
    }

    const float* weights_scale = (const float*)param.weight()->get_scale_data();
    if (outputs[0]->get_dtype() == AK_FLOAT) {
        conv_calibrate_int32_fp32(
                *outputs[0], *outputs[0], in_scale, weights_scale, *_ctx);
    } else if (outputs[0]->get_dtype() == AK_INT8) {
        // TODO THIS CAN BE A LOT OF WASTE OF PERF.
        conv_calibrate_int32_fp32(
                int8_output, int8_output, in_scale, weights_scale, *_ctx);

        std::vector<float> out_scale_v = outputs[0]->get_scale();
        if (out_scale_v.size() != 1) {
            LOG(FATAL) << "out scale set error, only support 1 scale for now!!! scale = "
                       << out_scale_v.size();
        }
        float out_scale = out_scale_v[0];
        conv_calibrate_fp32_int8_c4(*outputs[0], int8_output, out_scale, *_ctx);
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
        weights_temp.re_alloc(target_weights.valid_shape(), AK_INT8);
        weights_host.copy_from(target_weights);

        convert_weights_to_direct(weights_temp, weights_host, *_ctx);
        target_weights.set_dtype(AK_INT8);
        target_weights.re_alloc(target_weights.valid_shape(), AK_INT8);
        target_weights.set_layout(Layout_NCHW_C4);
        target_weights.copy_from(weights_temp);
        target_weights.set_scale(weights_temp.get_scale());

        if (target_bias.valid_size() > 0) {
            Tensor<NVHX86> bias_fp32_host;
            Tensor<NVHX86> bias_int32_host;
            bias_fp32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_int32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_fp32_host.copy_from(target_bias);
            convert_bias_host(bias_int32_host, bias_fp32_host, _in_scale,
                              target_weights.get_scale(), *_ctx);
            target_bias.copy_from(bias_int32_host);
        }
        return SaberSuccess;

    } else if (arch_check
        && target_weights.channel() % 4 == 0
        && target_weights.num() % 4 == 0) {

        // prepare int8 memory
        Tensor<NVHX86> weights_fp32_host;
        Tensor<NVHX86> weights_int8_host;
        weights_fp32_host.re_alloc(target_weights.valid_shape(), AK_FLOAT);
        weights_int8_host.re_alloc(target_weights.valid_shape(), AK_INT8);
        weights_int8_host.set_layout(Layout_NCHW_C4);
        weights_fp32_host.copy_from(target_weights);
        convert_weights_to_nchw_c4_host(weights_int8_host, weights_fp32_host, *_ctx);
        // Open this will be an inplace trans

        target_weights.set_dtype(AK_INT8);
        target_weights.re_alloc(target_weights.valid_shape(), AK_INT8);
        target_weights.set_layout(Layout_NCHW_C4);
        target_weights.copy_from(weights_int8_host);
        target_weights.set_scale(weights_int8_host.get_scale());
        if (target_bias.valid_size() > 0) {
            Tensor<NVHX86> bias_fp32_host;
            Tensor<NVHX86> bias_int32_host;
            bias_fp32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_int32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_fp32_host.copy_from(target_bias);
            convert_bias_host(bias_int32_host, bias_fp32_host, _in_scale,
                              target_weights.get_scale(), *_ctx);
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
