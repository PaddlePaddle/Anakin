
#include "saber/funcs/impl/cuda/saber_conv_winograd.h"
#include "saber/funcs/calibrate.h"
#include "saber_conv.h"

namespace anakin {
namespace saber {
template <>
SaberStatus SaberWinogradConv<AK_FLOAT>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();    
    const float* weight_ptr = (const float *) param.weight()->data();
    const float* bias_data = nullptr;
    if (param.bias()->size() > 0) {
        bias_data = (const float*)param.bias()->data();
    }

    if (param.activation_param.has_active)
    {
        if (param.activation_param.active == Active_relu)
        {
            winograd_conv_relu((const float *) inputs[0]->data(),
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
                      param.weight()->height(),
                      param.weight()->width(),
                      param.pad_h,
                      param.pad_w,
                      param.stride_h,
                      param.stride_w,
                      param.dilation_h,
                      param.dilation_w,
                      param.group,
                      param.alpha,
                      param.beta,
                      //nullptr,
                      this->_ctx->get_compute_stream()); 
        CUDA_CHECK(cudaGetLastError());
        return SaberSuccess;
        }
    }  
    winograd_conv((const float *) inputs[0]->data(),
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
                      param.weight()->height(),
                      param.weight()->width(),
                      param.pad_h,
                      param.pad_w,
                      param.stride_h,
                      param.stride_w,
                      param.dilation_h,
                      param.dilation_w,
                      param.group,
                      param.alpha,
                      param.beta,
                      //nullptr,
                      this->_ctx->get_compute_stream()); 
    if (this->_saber_act != nullptr) {
        this->_saber_act->dispatch(outputs, outputs, param.activation_param);
    }
    CUDA_CHECK(cudaGetLastError());
    return SaberSuccess;
}

template <>
SaberStatus SaberWinogradConv<AK_INT8>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {
    return SaberUnImplError;
}

template <>
SaberStatus SaberWinogradConv<AK_HALF>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {
    return SaberUnImplError;
}

}
}
