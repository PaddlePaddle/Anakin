
#include "saber/funcs/impl/cuda/vender_conv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus VenderConv2D<NV, AK_FLOAT>::\
    create(const std::vector<Tensor<NV> *>& inputs,
           std::vector<Tensor<NV> *>& outputs,
           ConvParam<NV>& param, Context<NV>& ctx) {

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
    int kernel_h = param.weight()->height();
    int kernel_w = param.weight()->width();
    int filter_dim_a[] = {output_channel,
                          input_channel / param.group, kernel_h, kernel_w};

    cudnn::setNDFilterDesc<OpDataType>(&_filter_desc,
                                       param.weight()->dims(), filter_dim_a, CUDNN_TENSOR_NCHW);

    Shape in_stride = inputs[0]->get_stride();
    Shape out_stride = outputs[0]->get_stride();

    int dim_a[] = {input_num, input_channel,
                   input_height, input_width};
    int dim_b[] = {input_num, output_channel,
                   output_height, output_width};

    cudnn::setTensorNdDesc<InDataType >(&_input_descs,
                                        inputs[0]->dims(), dim_a, &in_stride[0]);

    cudnn::setTensorNdDesc<OutDataType>(&_output_descs,
                                        outputs[0]->dims(), dim_b, &out_stride[0]);

    int pad_a[] = {param.pad_h, param.pad_w};
    int filter_stride_a[] = {param.stride_h, param.stride_w};
    int dilation_a[] = {param.dilation_h, param.dilation_w};

    cudnn::setConvolutionNdDesc<OpDataType >(&_conv_descs,
                                             inputs[0]->dims() - 2, pad_a,
                                             filter_stride_a, dilation_a);

    // true: use tensor core
    // false: disable tensor core
    cudnn::set_math_type<OpDataType>(&_conv_descs, _use_tensor_core);
    cudnn::set_group_count<OpDataType>(&_conv_descs, param.group);

    // Get fastest implement of cudnn
    // set up algo and workspace size
    if (param.group == inputs[0]->channel() && inputs[0]->channel() == outputs[0]->channel()) {
        _fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    } else {
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
                _handle, _input_descs, _filter_desc, _conv_descs, _output_descs,
                _preference, _workspace_limit_bytes, &_fwd_algo));
    }

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            _handle, _input_descs, _filter_desc, _conv_descs, _output_descs,
            _fwd_algo, &_workspace_fwd_sizes));

    if (_workspace_fwd_sizes > _workspaceSizeInBytes) {
        _workspaceSizeInBytes = _workspace_fwd_sizes;
        if (_workspaceData != NULL) {
            cudaFree(_workspaceData);
        }
        cudaMalloc(&_workspaceData, _workspaceSizeInBytes);
        _workspace = reinterpret_cast<char*>(_workspaceData);
    }

    if (param.bias()->size() > 0) {
        int dim_bias[] = {1, output_channel, 1, 1};
        int stride_bias[] = {output_channel, 1, 1, 1};
        cudnn::setTensorNdDesc<OpDataType >(&_bias_desc,
                                            4, dim_bias, stride_bias);
    }
    return SaberSuccess;
}
template <>
SaberStatus VenderConv2D<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ConvParam<NV>& param) {

    const InDataType* in_data = (const InDataType*)inputs[0]->data();
    OutDataType* out_data = (OutDataType*)outputs[0]->mutable_data();
    const float* weight_data = (const float*) param.weight()->data();

    CUDNN_CHECK(cudnnConvolutionForward(_handle,
                                        cudnn::cudnnTypeWrapper<float>::kOne(),
                                        _input_descs, in_data,
                                        _filter_desc, weight_data,
                                        _conv_descs,  _fwd_algo, _workspace, _workspace_fwd_sizes,
                                        cudnn::cudnnTypeWrapper<float>::kZero(),
                                        _output_descs, out_data));

    if (param.bias()->size() > 0) {
        // add up bias.
        const float* bias_data = (const float*)param.bias()->data();
        CUDNN_CHECK(cudnnAddTensor(_handle,
                                   cudnn::cudnnTypeWrapper<float>::kOne(),
                                   _bias_desc, bias_data,
                                   cudnn::cudnnTypeWrapper<float>::kOne(),
                                   _output_descs, out_data));
    }
    return SaberSuccess;
}

}
}
