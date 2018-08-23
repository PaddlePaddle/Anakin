
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/calibrate.h"
#include "saber_conv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberConv2D<NV, AK_FLOAT>::\
    init(const std::vector<Tensor<NV> *>& inputs,
           std::vector<Tensor<NV> *>& outputs,
           ConvParam<NV>& param, Context<NV>& ctx) {
    this->_ctx = &ctx;

    if (param.stride_h == 1 &&
        param.stride_w == 1 &&
        param.weight()->height() == 3 &&
        param.weight()->width() == 3 && param.group == 1) {
        if (param.activation_param.has_active) {
            if (param.activation_param.active == Active_relu) {
                dispatch_func = winograd_conv_relu<float, OpDataType>;
            } else {
                _with_saber_act = true;
                dispatch_func = winograd_conv<float, OpDataType>;
            }
        } else {
            dispatch_func = winograd_conv<float, OpDataType>;
        }
    } else if (param.group == 1) {
        const int K = param.weight()->num();
        if (K % 4 == 0) {
            if (param.bias()->size() > 0 && !param.activation_param.has_active) {
                dispatch_func = direct_conv_bias_Kdivis4<float, OpDataType>;
            } else if (param.bias()->valid_size() > 0 && param.activation_param.active == Active_relu) {
                dispatch_func = direct_conv_bias_relu_Kdivis4<float, OpDataType>;
            } else {
                if (param.activation_param.has_active) {
                    // NOT SUPPORT conv relu fusion
                    _with_saber_act = true;
                }
                dispatch_func = direct_conv_Kdivis4<float, OpDataType>;
            }
        } else {
            if (param.bias()->size() > 0 && !param.activation_param.has_active) {
                dispatch_func = direct_conv_bias_Kindiv4<float, OpDataType>;
            } else if (param.bias()->valid_size() > 0 && param.activation_param.active == Active_relu) {
                dispatch_func = direct_conv_bias_relu_Kindiv4<float, OpDataType>;
            } else {
                if (param.activation_param.has_active) {
                    // NOT SUPPORT conv relu fusion
                    _with_saber_act = true;
                }
                dispatch_func = direct_conv_Kindiv4<float, OpDataType>;
            }
        }
    } else if (param.group == inputs[0]->channel() && param.group == outputs[0]->channel()) {
        if (param.activation_param.has_active) {
            if (param.activation_param.active == Active_relu) {
                if (param.bias()->size() > 0) {
                    depthwise_func = saber_depthwise_conv_act<float, true, true>;
                } else {
                    depthwise_func = saber_depthwise_conv_act<float, false, true>;
                }
            } else {
                if (param.bias()->size() > 0) {
                    depthwise_func = saber_depthwise_conv_act<float, true, false>;
                } else {
                    depthwise_func = saber_depthwise_conv_act<float, false, false>;
                }
                _with_saber_act = true;
            }
        } else if (param.bias()->size() > 0) {
            depthwise_func = saber_depthwise_conv_act<float, true, false>;
        } else {
            depthwise_func = saber_depthwise_conv_act<float, false, false>;
        }
    } else {
        return SaberUnImplError;
    }

    _kernel_height = param.weight()->height();
    _kernel_width = param.weight()->width();
    _use_k1s1p0 = true;
    _use_k1s1p0 = _use_k1s1p0 && (_kernel_height == 1);
    _use_k1s1p0 = _use_k1s1p0 && (_kernel_width == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.pad_h == 0);
    _use_k1s1p0 = _use_k1s1p0 && (param.pad_w == 0);
    _use_k1s1p0 = _use_k1s1p0 && (param.stride_h == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.stride_w == 1);
    _use_k1s1p0 = _use_k1s1p0 && (param.bias()->valid_size()>0);
    _use_k1s1p0 = _use_k1s1p0 && (param.activation_param.active == Active_relu);

    if (_use_k1s1p0) {
        return SaberSuccess;
    }

    if (_with_saber_act) {
        _saber_act = new SaberActivation<NV, AK_FLOAT>;
        _saber_act->init(outputs, outputs, param.activation_param, ctx);
    }

    conv_trans_weights<NV, NVHX86>(inputs, outputs, param, ctx, _in_place, &_weight_dev);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConv2D<NV, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<NV> *>& inputs,
             std::vector<Tensor<NV> *>& outputs,
             ConvParam<NV>& param) {
    //err code?
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    const float* bias_data = nullptr;
    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chout = outputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int in_stride = chin * win * hin;
    int out_stride = chout * wout * hout;
    if (param.bias()->size() > 0) {
        bias_data = (const float*)param.bias()->data();
    }
    if (param.group == inputs[0]->channel() && param.group == outputs[0]->channel()) {
        depthwise_func((const float*)inputs[0]->data(),
                       (float*)outputs[0]->mutable_data(),
                       inputs[0]->num(), inputs[0]->channel(),
                       inputs[0]->height(), inputs[0]->width(), outputs[0]->height(),
                       outputs[0]->width(), _kernel_width, _kernel_height, param.stride_w,
                       param.stride_h, param.pad_w, param.pad_h,
                       (const OpDataType*)param.weight()->data(), (const float*)bias_data,
                       this->_ctx->get_compute_stream());
    } else if (_use_k1s1p0){
        if (param.activation_param.has_active) {
            conv_gemm_k1s1p0<true>(num, in_stride, out_stride,
                                   (float*)outputs[0]->mutable_data(),
                                   (const float*)inputs[0]->data(),
                                   (const float*)param.weight()->data(),
                                   chout, chin, hin, win, bias_data,
                                   this->_ctx->get_compute_stream(), 1.f, 0.f);
        } else {
            conv_gemm_k1s1p0<false>(num, in_stride, out_stride,
                                    (float*)outputs[0]->mutable_data(),
                                    (const float*)inputs[0]->data(),
                                    (const float*)param.weight()->data(),
                                    chout, chin, hin, win, bias_data,
                                    this->_ctx->get_compute_stream(), 1.f, 0.f);
        }
        return SaberSuccess;
    } else {
        const float* weight_ptr = nullptr;
        if (_in_place) {
            weight_ptr = (const float *) param.weight()->data();
        } else {
            weight_ptr = (const float *) _weight_dev.data();
        }
        dispatch_func((const float *) inputs[0]->data(),
                      (float *) outputs[0]->mutable_data(),
                      weight_ptr,
                      bias_data,
                      inputs[0]->num(),
                      inputs[0]->channel(),
                      inputs[0]->height(),
                      inputs[0]->width(),
                      outputs[0]->channel(),
                      outputs[0]->height(),
                      outputs[0]->width(),
                      shape_in[1],
                      shape_in[2],
                      shape_in[3],
                      shape_out[1],
                      shape_out[2],
                      shape_out[3],
                      _kernel_height,
                      _kernel_width,
                      param.pad_h,
                      param.pad_w,
                      param.stride_h,
                      param.stride_w,
                      param.dilation_h,
                      param.dilation_w,
                      param.group,
                      param.alpha,
                      param.beta,
                      this->_ctx->get_compute_stream());
    }

    if (_with_saber_act) {
        _saber_act->dispatch(outputs, outputs, param.activation_param);
    }
    CUDA_CHECK(cudaGetLastError());
    return SaberSuccess;
}

template <>
SaberStatus SaberConv2D<NV, AK_INT8>::\
    init(const std::vector<Tensor<NV> *>& inputs,
         std::vector<Tensor<NV> *>& outputs,
         ConvParam<NV>& param, Context<NV>& ctx) {
    return SaberInvalidValue;
}

template <>
SaberStatus SaberConv2D<NV, AK_INT8>::\
    dispatch(const std::vector<Tensor<NV> *>& inputs,
             std::vector<Tensor<NV> *>& outputs,
             ConvParam<NV>& param) {

    return SaberInvalidValue;
}

}
}