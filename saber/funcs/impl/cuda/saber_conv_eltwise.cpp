
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/impl/cuda/saber_conv_eltwise.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
#include "saber/funcs/calibrate.h"
#include "saber_conv_eltwise.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberConvEltwise<NV, AK_FLOAT>::\
    init(const std::vector<Tensor<NV> *>& inputs,
            std::vector<Tensor<NV> *>& outputs,
            ConvEltwiseParam<NV>& param, Context<NV>& ctx) {

    _ctx = &ctx;
    _kernel_height = param.conv_param.weight()->height();
    _kernel_width = param.conv_param.weight()->width();

    _use_k1s1p0 = true;
    _use_k1s1p0 = _use_k1s1p0 && (_kernel_height == 1);
    _use_k1s1p0 = _use_k1s1p0 && (_kernel_width == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.pad_h == 0);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.pad_w == 0);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.stride_h == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.stride_w == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.bias()->valid_size()>0);
    _use_k1s1p0 = _use_k1s1p0 && (!param.conv_param.activation_param.has_active);

    if (_use_k1s1p0) {
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConvEltwise<NV, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<NV> *>& inputs,
            std::vector<Tensor<NV> *>& outputs,
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
    } else {
        return SaberUnImplError;
    }
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberConvEltwise, ConvEltwiseParam, NV, AK_INT16);
DEFINE_OP_TEMPLATE(SaberConvEltwise, ConvEltwiseParam, NV, AK_INT8);
}
}