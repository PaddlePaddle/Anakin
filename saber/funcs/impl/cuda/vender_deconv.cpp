
#include "saber/funcs/impl/cuda/vender_deconv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus VenderDeconv2D<NV, AK_FLOAT>::create(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV>& ctx) {

    if (!(&ctx == this->_ctx)) {
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

    int filter_dim_a[] = {input_channel, output_channel / param.group, \
                          kernel_h, kernel_w
                         };

    cudnn::setNDFilterDesc<OpDataType>(&_filter_desc,
                                       param.weight()->dims(),
                                       filter_dim_a, CUDNN_TENSOR_NCHW);

    Shape in_stride = inputs[0]->get_stride();
    Shape out_stride = outputs[0]->get_stride();

    int dim_a[] = {input_num, input_channel,
                   input_height, input_width
                  };

    int dim_b[] = {input_num, output_channel,
                   output_height, output_width
                  };

    cudnn::setTensorNdDesc<OpDataType>(&_input_descs,
                                       inputs[0]->dims(), dim_a, &in_stride[0]);

    cudnn::setTensorNdDesc<OpDataType>(&_output_descs,
                                       outputs[0]->dims(), dim_b, &out_stride[0]);

    int pad_a[] = {param.pad_h, param.pad_w};
    int stride_a[] = {param.stride_h, param.stride_w};
    int dilation_a[] = {param.dilation_h, param.dilation_w};

    //cudnn::setConvolutionNdDesc<OpDtype >(&_conv_descs, \
    inputs[0]->dims() - 2, pad_a, \
    stride_a, dilation_a);
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(_conv_descs, \
                pad_a[0], pad_a[1], \
                stride_a[0], stride_a[1], \
                dilation_a[0], dilation_a[1], \
                CUDNN_CROSS_CORRELATION, \
                cudnn::cudnnOpWrapper<OpDataType>::type));

    // true: use tensor core
    // false: disable tensor core
    cudnn::set_math_type<OpDataType>(&_conv_descs, _use_tensor_core);
    cudnn::set_group_count<OpDataType>(&_conv_descs, param.group);

    // Get fastest implement of cudnn
    // set up algo and workspace size
    //CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(_handle, \
    _filter_desc, _input_descs, _conv_descs, _output_descs, \
    _preference, _workspace_limit_bytes, &_bwd_algo));
    _bwd_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(_handle,
                _filter_desc, _input_descs, _conv_descs, _output_descs,
                _bwd_algo, &_workspace_bwd_sizes));

    if (_workspace_bwd_sizes > _workspaceSizeInBytes) {
    _workspaceSizeInBytes = _workspace_bwd_sizes;

    if (workspaceData != NULL) {
            CUDA_CHECK(cudaFree(workspaceData));
        }

        CUDA_CHECK(cudaMalloc(&workspaceData, _workspaceSizeInBytes));
        workspace = reinterpret_cast<char*>(workspaceData);
    }

    if (param.bias()->size() > 0) {
    int dim_bias[] = {1, output_channel, 1, 1};
        int stride_bias[] = {output_channel, 1, 1, 1};
        cudnn::setTensorNdDesc<OpDataType>(&_bias_desc,
                                           4, dim_bias, stride_bias);
    }

    if (_use_saber_act) {
    _saber_act.create(inputs, outputs, param.activation_param, ctx);
    }

    return SaberSuccess;
}

template <>
SaberStatus VenderDeconv2D<NV, AK_FLOAT>::init(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV>& ctx) {

    // ---- init cudnn resources ----

    _workspaceSizeInBytes = 0;
    workspaceData = NULL;

    _workspace_bwd_sizes = 0;

    this->_ctx = &ctx;
    // ---- get cuda resources ----

    cudaStream_t cuda_stream;
    cuda_stream = ctx.get_compute_stream();

    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));
    workspace = NULL;
    int in_channels = inputs[0]->channel();

    // ---- create cudnn Descs ----
    cudnn::createFilterDesc<OpDataType>(&_filter_desc);
    cudnn::createTensorDesc<Tensor<NV> >(&_input_descs);
    cudnn::createTensorDesc<Tensor<NV> >(&_output_descs);
    cudnn::createConvolutionDesc<OpDataType>(&_conv_descs);
    cudnn::createTensorDesc<OpDataType>(&_bias_desc);

    if (param.activation_param.has_active) {
        _saber_act.init(inputs, outputs, param.activation_param, ctx);
        _use_saber_act = true;
    }

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderDeconv2D<NV, AK_FLOAT>::dispatch(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param) {
    const float* input_data = (const float*)inputs[0]->data();
    float* output_data = (float*)outputs[0]->mutable_data();
    const float* weight_data = (const float*) param.weight()->data();

    CUDNN_CHECK(cudnnConvolutionBackwardData(_handle,
                cudnn::cudnnTypeWrapper<float>::kOne(),
                _filter_desc, weight_data,
                _input_descs, input_data,
                _conv_descs,  _bwd_algo, workspace, _workspace_bwd_sizes,
                cudnn::cudnnTypeWrapper<float>::kZero(),
                _output_descs, output_data));

    if (param.bias()->valid_size() > 0) {
        const float* bias_data;
        bias_data = (const float*)param.bias()->data();
        CUDNN_CHECK(cudnnAddTensor(
                        _handle,
                        cudnn::cudnnTypeWrapper<float>::kOne(),
                        _bias_desc,
                        bias_data,
                        cudnn::cudnnTypeWrapper<float>::kOne(),
                        _output_descs,
                        output_data));
    }

    if (_use_saber_act) {
        _saber_act.dispatch(outputs, outputs, param.activation_param);
    }

    return SaberSuccess;
}

template class VenderDeconv2D<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderDeconv2D, ConvParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderDeconv2D, ConvParam, NV, AK_INT8);

}
}
