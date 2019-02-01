
#include "saber/funcs/impl/cuda/saber_conv_gemmlike.h"
#include "saber/funcs/calibrate.h"
#include "saber_conv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberGemmLikeConv<AK_FLOAT>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {

    return SaberSuccess;
}

template <>
SaberStatus SaberGemmLikeConv<AK_FLOAT>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberGemmLikeConv<AK_FLOAT>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {

    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chout = outputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int in_stride = chin * win * hin;
    int out_stride = chout * wout * hout;

    const float* bias_data = nullptr;
    if (param.bias()->size() > 0) {
        bias_data = (const float*)param.bias()->data();
    }

    if (param.activation_param.has_active) {
        if (param.activation_param.active == Active_relu) {
            conv_gemm_k1s1p0<true>(num, in_stride, out_stride,
                                   (float*)outputs[0]->mutable_data(),
                                   (const float*)inputs[0]->data(),
                                   (const float*)param.weight()->data(),
                                   chout, chin, hin, win, bias_data,
                                   this->_ctx->get_compute_stream(), 1.f, 0.f);
            CUDA_CHECK(cudaGetLastError());
            return SaberSuccess;
        }
    }

    conv_gemm_k1s1p0<false>(num, in_stride, out_stride,
                            (float*)outputs[0]->mutable_data(),
                            (const float*)inputs[0]->data(),
                            (const float*)param.weight()->data(),
                            chout, chin, hin, win, bias_data,
                            this->_ctx->get_compute_stream(), 1.f, 0.f);

    if (this->_saber_act != nullptr) {
        this->_saber_act->dispatch(outputs, outputs, param.activation_param);
    }
    CUDA_CHECK(cudaGetLastError());
    return SaberSuccess;
}

template <>
SaberStatus SaberGemmLikeConv<AK_INT8>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {

    return SaberSuccess;
}

template <>
SaberStatus SaberGemmLikeConv<AK_INT8>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {
    _ctx = &ctx;
    if (outputs[0]->get_dtype() == AK_FLOAT) {
        if (param.activation_param.has_active) {
            if (param.activation_param.active == Active_relu) {
                _int8_func = conv_igemm_k1s1p0<true>;
            } else {
                _int8_func = conv_igemm_k1s1p0<false>;
            }
        } else {
            _int8_func = conv_igemm_k1s1p0<false>;
        }
    } else if (outputs[0]->get_dtype() == AK_INT8) {
        if (param.activation_param.has_active || _use_act) {
            if (param.activation_param.active == Active_relu || _use_act) {
                _int8_func = conv_igemm_s8s8_k1s1p0<true>;
            } else {
                _int8_func = conv_igemm_s8s8_k1s1p0<false>;
            }
        } else {
            _int8_func = conv_igemm_s8s8_k1s1p0<false>;
        }
    }

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberGemmLikeConv<AK_INT8>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {
    int num = inputs[0]->num();
    int chin = inputs[0]->channel() / 4;
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chout = outputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int in_stride = chin * win * hin;
    int out_stride = chout * wout * hout;

    const void* bias_data = nullptr;
    if (param.bias()->size() > 0) {
        bias_data = (const void*)param.bias()->data();
    }
    float alpha = 1.f;
    if (param.weight()->get_scale().size() == 1) {
        CHECK_GE(inputs[0]->get_scale().size(), 1);
        alpha = inputs[0]->get_scale()[0] * param.weight()->get_scale()[0];
    }
    if (outputs[0]->get_dtype() == AK_INT8) {
        CHECK_GE(outputs[0]->get_scale().size(), 1);
        alpha /= outputs[0]->get_scale()[0];
    }
    float beta = param.beta;
//    LOG(INFO) << beta;
    _int8_func(num, in_stride, out_stride,
            (void*)outputs[0]->mutable_data(),
            (const void*)inputs[0]->data(),
            (const void*)param.weight()->data(),
            chout, chin, hin, win, bias_data,
            this->_ctx->get_compute_stream(), alpha, beta, 32);

//    if (this->_saber_act != nullptr) {
//        this->_saber_act->dispatch(outputs, outputs, param.activation_param);
//    }
    CUDA_CHECK(cudaGetLastError());
    return SaberSuccess;
}

}
}
