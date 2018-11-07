
#include "saber/funcs/impl/cuda/vender_lrn.h"

namespace anakin {
namespace saber {

// FP32 part
template <>
SaberStatus VenderLrn<NV, AK_FLOAT>::\
create(const std::vector<Tensor<NV> *>& inputs,
       std::vector<Tensor<NV> *>& outputs,
       LrnParam<NV>& param, Context<NV>& ctx) {
    if (&ctx != this->_ctx) {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }

        this->_ctx = &ctx;
        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();
        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));
    }

    int input_num = inputs[0]->num();
    int input_channel = inputs[0]->channel();
    int input_height = inputs[0]->height();
    int input_width = inputs[0]->width();
    int output_channel = outputs[0]->channel();
    int output_height = outputs[0]->height();
    int output_width = outputs[0]->width();

    Shape in_stride = inputs[0]->get_stride();
    Shape out_stride = outputs[0]->get_stride();

    int dim_a[] = {input_num, input_channel,
                   input_height, input_width
                  };
    int dim_b[] = {input_num, output_channel,
                   output_height, output_width
                  };

    cudnn::setTensorNdDesc<float >(&_input_descs,
                                   inputs[0]->dims(), dim_a, &in_stride[0]);

    cudnn::setTensorNdDesc<float>(&_output_descs,
                                  outputs[0]->dims(), dim_b, &out_stride[0]);


    cudnn::setLrnDesc<OpDataType>(&_lrn_descs,
                                  param.local_size, param.alpha * param.local_size,
                                  param.beta, param.k);
    return SaberSuccess;
}

template <>
SaberStatus VenderLrn<NV, AK_FLOAT>::\
init(const std::vector<Tensor<NV> *>& inputs,
     std::vector<Tensor<NV> *>& outputs,
     LrnParam<NV>& param, Context<NV>& ctx) {

    // ---- init cudnn resources ----
    this->_ctx = &ctx;
    // ---- get cuda resources ----
    cudaStream_t cuda_stream;
    cuda_stream = ctx.get_compute_stream();
    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

    // ---- create cudnn Descs ----
    cudnn::createTensorDesc<OpDataType>(&_input_descs);
    cudnn::createTensorDesc<OpDataType>(&_output_descs);
    cudnn::createLrnDesc<OpDataType>(&_lrn_descs);

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderLrn<NV, AK_FLOAT>::dispatch(
    const std::vector<Tensor<NV>*>& inputs,
    std::vector<Tensor<NV>*>& outputs,
    LrnParam<NV>& param) {
    CHECK(param.norm_region == saber::ACROSS_CHANNELS) << "vender lrn can not support within channel";
    const float* in_data = (const float*)inputs[0]->data();
    float* out_data = (float*)outputs[0]->mutable_data();
    auto alpha = 1.0f;
    auto beta = 0.f;

    CUDNN_CHECK(cudnnLRNCrossChannelForward(_handle,
                                            _lrn_descs,
                                            _lrn_mode,
                                            &(alpha),
                                            _input_descs,
                                            in_data,
                                            &(beta),
                                            _output_descs,
                                            out_data));
}



template class VenderLrn<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderLrn, LrnParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderLrn, LrnParam, NV, AK_INT8);
}
}
