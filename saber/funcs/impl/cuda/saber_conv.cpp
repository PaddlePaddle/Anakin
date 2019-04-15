
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
SaberStatus SaberConv2D<NV, AK_INT8>::trans_weights(Tensor<NV> &target_weights,
                                                     Tensor<NV> &target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
                                                     int stride_h, int stride_w, int group) {
    return SaberSuccess;
};

DEFINE_OP_TEMPLATE(SaberConv2D, ConvParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberConv2D, ConvParam, NV, AK_HALF);
}
}
