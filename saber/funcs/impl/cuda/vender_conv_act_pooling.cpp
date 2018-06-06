#include "saber/funcs/impl/cuda/vender_conv_act_pooling.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <>
SaberStatus VenderConv2DActPooling<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
    create(const std::vector<DataTensor_in *>& inputs,
            std::vector<DataTensor_out *>& outputs,
            ConvActivePoolingParam<OpTensor>& param, Context<NV> &ctx) {

    if (!(ctx == this->_ctx)) {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
        this->_ctx = ctx;

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
    {
        _inner_shape = inputs[0]->shape();
        _inner_shape[0] = input_num;
        _inner_shape[1] = param.conv_param.weight()->num();

        int kernel_exten = param.conv_param.dilation_h *
                           (param.conv_param.weight()->height() - 1) + 1;
        int output_dim = (input_height + 2 * param.conv_param.pad_h - kernel_exten)
                         / param.conv_param.stride_h + 1;
        _inner_shape[2] = output_dim;
        kernel_exten = param.conv_param.dilation_w *
                           (param.conv_param.weight()->width() - 1) + 1;
        output_dim = (input_width + 2 * param.conv_param.pad_w - kernel_exten)
                         / param.conv_param.stride_w + 1;
        _inner_shape[3] = output_dim;
        _inner_tensor.re_alloc(_inner_shape);
    }

    int kernel_h = param.conv_param.weight()->height();
    int kernel_w = param.conv_param.weight()->width();

    int filter_dim_a[] = {output_channel,
                          input_channel / param.conv_param.group,
                          kernel_h, kernel_w};

    cudnn::setNDFilterDesc<OpDataType>(&_filter_desc,
                                    param.conv_param.weight()->dims(),
                                    filter_dim_a, CUDNN_TENSOR_NCHW);

    Shape in_stride = inputs[0]->get_stride();
    Shape inner_stride = _inner_tensor.get_stride();
    Shape out_stride = outputs[0]->get_stride();

    int dim_a[] = {input_num, input_channel,
                   input_height, input_width};

    int dim_inner[] = {_inner_shape[0], _inner_shape[1],
                        _inner_shape[2], _inner_shape[3]};

    int dim_b[] = {input_num, output_channel,
                   output_height, output_width};
    cudnn::setTensorNdDesc<InDataType >(&_input_descs,
                                       inputs[0]->dims(), dim_a, &in_stride[0]);
    cudnn::setTensorNdDesc<InDataType >(&_inner_descs,
                                       4, dim_inner,
                                       &inner_stride[0]);
    cudnn::setTensorNdDesc<InDataType>(&_output_descs,
                                      outputs[0]->dims(), dim_b, &out_stride[0]);
    int pad_a[] = {param.conv_param.pad_h, param.conv_param.pad_w};
    int filter_stride_a[] = {param.conv_param.stride_h, param.conv_param.stride_w};
    int dilation_a[] = {param.conv_param.dilation_h, param.conv_param.dilation_w};

    cudnn::setConvolutionNdDesc<OpDataType >(&_conv_descs,
                                          inputs[0]->dims() - 2, pad_a,
                                          filter_stride_a, dilation_a);
    // set activation descriptor
    if (param.has_activation) {
        cudnn::set_activation_des<OpDataType>(&_active_descs, param.activation_param.active);
    }
    if (param.has_pooling) {
        int windowHeight[] = {param.pooling_param.window_h,
                              param.pooling_param.window_w};
        int padding[] = {param.pooling_param.pad_h,
                         param.pooling_param.pad_w};
        int stride[] = {param.pooling_param.stride_h,
                        param.pooling_param.stride_w};

        cudnn::set_nd_pooling_des<OpDataType >(&_pooling_descs,
                                            param.pooling_param.pooling_type,
                                            _inner_tensor.dims() - 2,
                                            windowHeight,
                                            padding,stride);
    }
    // true: use tensor core
    // false: disable tensor core
    cudnn::set_math_type<OpDataType>(&_conv_descs, _use_tensor_core);
    cudnn::set_group_count<OpDataType>(&_conv_descs, param.conv_param.group);

    // Get fastest implement of cudnn
    // set up algo and workspace size
    if (param.conv_param.group == inputs[0]->channel() && \
        inputs[0]->channel() == outputs[0]->channel()) {
        _fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;//CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    } else {
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(_handle, \
            _input_descs, _filter_desc, _conv_descs, _inner_descs, \
            _preference, _workspace_limit_bytes, &_fwd_algo));
    }

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_handle,
                                                        _input_descs, _filter_desc,
                                                        _conv_descs, _inner_descs,
                                                        _fwd_algo, &_workspace_fwd_sizes));

    if (_workspace_fwd_sizes > _workspaceSizeInBytes) {
        _workspaceSizeInBytes = _workspace_fwd_sizes;
        if (_workspaceData != NULL) {
            cudaFree(_workspaceData);
        }
        cudaMalloc(&_workspaceData, _workspaceSizeInBytes);
        _workspace = reinterpret_cast<char*>(_workspaceData);
    }

    if (param.conv_param.bias()->size()> 0) {
        int dim_bias[] = {1, output_channel, 1, 1};
        int stride_bias[] = {output_channel, 1, 1, 1};

        cudnn::setTensorNdDesc<OpDataType >(&_bias_desc,
                                         4, dim_bias, stride_bias);
    }
    return SaberSuccess;
}
template <>
SaberStatus VenderConv2DActPooling<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
    dispatch(const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                ConvActivePoolingParam<OpTensor>& param) {

    const InDataType *in_data = (const InDataType*)inputs[0]->data();
    InDataType *inner_data = (InDataType*)_inner_tensor.mutable_data();
    InDataType *out_data = (InDataType*)outputs[0]->mutable_data();

    const float *weight_data = (const float *) param.conv_param.weight()->data();
    if (param.has_activation == false) {
        CUDNN_CHECK(cudnnConvolutionForward(_handle,
                                            cudnn::cudnnTypeWrapper<float>::kOne(),
                                            _input_descs, in_data,
                                            _filter_desc, weight_data,
                                            _conv_descs,  _fwd_algo, _workspace, _workspace_fwd_sizes,
                                            cudnn::cudnnTypeWrapper<float>::kZero(),
                                            _inner_descs, inner_data
        ));
        if (param.conv_param.bias()->size() > 0) {
            // add up bias.
            const float * bias_data = (const float*)param.conv_param.bias()->data();
            CUDNN_CHECK(cudnnAddTensor(_handle,
                                       cudnn::cudnnTypeWrapper<float>::kOne(),
                                       _bias_desc, bias_data,
                                       cudnn::cudnnTypeWrapper<float>::kOne(),
                                       _inner_descs, inner_data));
        }
        CUDNN_CHECK(cudnnPoolingForward(_handle, _pooling_descs,
                                        cudnn::cudnnTypeWrapper<InDataType>::kOne(),
                                        _inner_descs, inner_data,
                                        cudnn::cudnnTypeWrapper<InDataType>::kZero(),
                                        _output_descs, out_data
        ));
        return SaberSuccess;
    }

    if (param.conv_param.bias()->size() > 0) {
        const float * bias_data = (const float*)param.conv_param.bias()->data();
        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(_handle,
                                                          cudnn::cudnnTypeWrapper<float>::kOne(),
                                                          _input_descs, in_data,
                                                          _filter_desc, weight_data,
                                                          _conv_descs, _fwd_algo,
                                                          _workspace, _workspace_fwd_sizes,
                                                          cudnn::cudnnTypeWrapper<float>::kZero(),
                                                          _inner_descs, inner_data,
                                                          _bias_desc,  bias_data,
                                                          _active_descs, _inner_descs, inner_data));

        CUDNN_CHECK(cudnnPoolingForward(_handle, _pooling_descs,
                                        cudnn::cudnnTypeWrapper<InDataType>::kOne(),
                                        _inner_descs, inner_data,
                                        cudnn::cudnnTypeWrapper<InDataType>::kZero(),
                                        _output_descs, out_data
        ));

    } else {

        CUDNN_CHECK(cudnnConvolutionForward(_handle,
                                            cudnn::cudnnTypeWrapper<float>::kOne(),
                                            _input_descs, in_data,
                                            _filter_desc, weight_data,
                                            _conv_descs, _fwd_algo,
                                            _workspace, _workspace_fwd_sizes,
                                            cudnn::cudnnTypeWrapper<float>::kZero(),
                                            _inner_descs, inner_data
        ));

        CUDNN_CHECK(cudnnActivationForward(_handle, _active_descs,
                                           cudnn::cudnnTypeWrapper<InDataType>::kOne(),
                                           _inner_descs, inner_data,
                                           cudnn::cudnnTypeWrapper<InDataType>::kZero(),
                                           _inner_descs, inner_data
        ));
        CUDNN_CHECK(cudnnPoolingForward(_handle, _pooling_descs,
                                        cudnn::cudnnTypeWrapper<InDataType>::kOne(),
                                        _inner_descs, inner_data,
                                        cudnn::cudnnTypeWrapper<InDataType>::kZero(),
                                        _output_descs, out_data
        ));
    }
    return SaberSuccess;
}
}
}
