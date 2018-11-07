
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
SaberStatus SaberConv2D<NV, AK_FLOAT>::init(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV>& ctx) {
    this->_ctx = &ctx;
    int generate_arch = Env<NV>::cur_env()[ctx.get_device_id()]._info._generate_arch;
    bool arch_check = generate_arch == 61;

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

        if (use_k1s1p0) {
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
SaberStatus SaberConv2D<NV, AK_FLOAT>::trans_weights(Tensor<NV>& target_weights,
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
SaberStatus SaberConv2D<NV, AK_INT8>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {

    _impl = new SaberDirectConv<AK_INT8>;

    _impl->init(inputs, outputs, param, ctx);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConv2D<NV, AK_HALF>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {

    return SaberUnImplError;
}

template <>
SaberStatus SaberConv2D<NV, AK_INT8>::trans_weights(Tensor<NV>& target_weights,
        Tensor<NV>& target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberUnImplError;
}

template <>
SaberStatus SaberConv2D<NV, AK_HALF>::trans_weights(Tensor<NV>& target_weights,
        Tensor<NV>& target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberUnImplError;

}

}
}
