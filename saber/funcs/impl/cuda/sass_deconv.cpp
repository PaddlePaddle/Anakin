
#include "saber/funcs/impl/cuda/sass_deconv.h"
#include "saber/funcs/impl/impl_macro.h"
#include "sass_funcs.h"

namespace anakin {

namespace saber {

template <>
SaberStatus SassDeconv<NV, AK_FLOAT>::create(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {

    if (!_extern_trans) {
        int in_channel = inputs[0]->channel();
        int out_channel = outputs[0]->channel();
        scale_to_new_tensor_k4_s2_p1_deconv<Tensor<NV>, Tensor<NVHX86>, 4>(*param.mutable_weight(),
                in_channel, out_channel, _in_place, &_weight_dev);
    }

    return SaberSuccess;
}

template <>
SaberStatus SassDeconv<NV, AK_FLOAT>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SassDeconv<NV, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();
    const float* din = (const float*)inputs[0]->data();
    float* dout = (float*)outputs[0]->mutable_data();

    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int num = inputs[0]->num();
    int ch_in = inputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int ch_out = outputs[0]->channel();

    // transform
    // PAY ATTENTION!!!!
    // The shape of weights is suppose to be {in_channel, out_channel, kernel_size, kernel_size};
    // but caffe is reshaped their shape as {out, in, kernel_size, kernel_size}
    // so this need to reshaped like caffe as {out_channel, in_channel, kernel_size, kernel_size}
    // The param of transform_weights_deconv:
    // int in_channel  : the in_channel of the img(where loop running on!)
    //                   this param must be the seam with img in_channel,
    //
    // int out_channel : the real output filter num(as much as you can, this is the proto param)
    //
    // const float *
    //     weights_src : the real data is orgnized as
    //                   (in_channel, out_channel, kernel_size, kernel_size)
    // const float *
    //     XX_out      : the output data is orgnized as
    //                   (out_channel, in_channel, kernel_size, kernel_size)
    //                   just like normal convolution weights
    //    weights_transform_dev.copy_from(weights_origin_host);

    const float* bias_data = (param.bias()->valid_size() > 0) ?
                             (const float*)param.bias()->data() : NULL;

    const float* weights_data = nullptr;

    if (_in_place) {
        weights_data = (const float*) param.weight()->data();
    } else {
        weights_data = (const float*) _weight_dev.data();
    }

    if (param.activation_param.has_active && !_with_saber_act) {
        ker_deconv_implicit_gemm_k4_s2_p1_32x32_relu(dout, din,
                weights_data, bias_data,
                num,
                hin, win, hout, wout,
                ch_in, ch_out, stream);
    } else {
        ker_deconv_implicit_gemm_k4_s2_p1_16x64(dout, din,
                                                weights_data, bias_data,
                                                num,
                                                hin, win, hout, wout,
                                                ch_in, ch_out, stream);
    }

    return SaberSuccess;
}

template class SassDeconv<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SassDeconv, ConvParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SassDeconv, ConvParam, NV, AK_INT8);
}
}