
#include "vender_softmax.h"
namespace anakin {
namespace saber {
template <>
SaberStatus VenderSoftmax<NV, AK_FLOAT>::create(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SoftmaxParam<NV>& param, Context<NV>& ctx) {

    if (!inputs[0]->is_continue_mem() || !outputs[0]->is_continue_mem()) {
        //! unsupported type for cudnn
        return SaberInvalidValue;
    }


    Shape shape_in = inputs[0]->valid_shape();

    if (!(ctx == *(this->_ctx))) {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }

        this->_ctx = &ctx;
        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();
        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));
    }

    int outer_num = inputs[0]->count(0, param.axis);
    int inner_num = inputs[0]->count(param.axis + 1, inputs[0]->dims());

    int N = outer_num;
    int K = inputs[0]->valid_shape()[param.axis];
    int H = inner_num;
    int W = 1;

    const int stride_w = 1;
    const int stride_h = W * stride_w;
    const int stride_c = H * stride_h;
    const int stride_n = K * stride_c;
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(_input_desc, \
                cudnn::cudnnOpWrapper<OpDataType>::type, \
                N, K, H, W, stride_n, stride_c, stride_h, stride_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(_output_desc, \
                cudnn::cudnnOpWrapper<OpDataType>::type, \
                N, K, H, W, stride_n, stride_c, stride_h, stride_w));

    _setup = true;
    return SaberSuccess;
}

/**
 * \brief initial all cudnn resources here
 * @param inputs
 * @param outputs
 * @param param
 * @param ctx
 */
template <>
SaberStatus VenderSoftmax<NV, AK_FLOAT>::init(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SoftmaxParam<NV>& param, Context<NV>& ctx) {

    // ---- init cudnn resources ----

    this->_ctx = &ctx;
    // ---- get cuda resources ----

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

    // ---- create cudnn Descs ----
    cudnn::createTensorDesc<OpDataType>(&_input_desc);
    cudnn::createTensorDesc<OpDataType>(&_output_desc);

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderSoftmax<NV, AK_FLOAT>::dispatch(const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SoftmaxParam<NV>& param) {
    cudaStream_t stream = this->_ctx->get_compute_stream();
    const OpDataType* input_data = (const OpDataType*)inputs[0]->data();
    OpDataType* output_data = (OpDataType*)outputs[0]->mutable_data();
    CUDNN_CHECK(cudnnSoftmaxForward(_handle, CUDNN_SOFTMAX_ACCURATE, \
                                    CUDNN_SOFTMAX_MODE_CHANNEL, cudnn::cudnnTypeWrapper<OpDataType>::kOne(), _input_desc, input_data, \
                                    cudnn::cudnnTypeWrapper<OpDataType>::kZero(), _output_desc, output_data));
    //outputs[0]->record_event(stream);
    return SaberSuccess;
}


DEFINE_OP_TEMPLATE(VenderSoftmax, SoftmaxParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderSoftmax, SoftmaxParam, NV, AK_INT8);

}
}