#include "saber/funcs/impl/cuda/vender_conv_act.h"
#include "saber/funcs/impl/cuda/saber_activation.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <>
SaberStatus VenderConv2DAct<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
    create(const std::vector<DataTensor_in *>& inputs,
            std::vector<DataTensor_out *>& outputs,
            ConvActiveParam<OpTensor>& param, Context<NV>& ctx) {

    if (_use_saber_act) {
        _saber_act.create(inputs, outputs, param.activation_param, ctx);
    }
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

    int kernel_h = param.conv_param.weight()->height();
    int kernel_w = param.conv_param.weight()->width();

    int filter_dim_a[] = {output_channel,
                          input_channel / param.conv_param.group,
                          kernel_h, kernel_w};

    cudnn::setNDFilterDesc<OpDataType>(&_filter_desc,
                                    param.conv_param.weight()->dims(),
                                    filter_dim_a, CUDNN_TENSOR_NCHW);

    Shape in_stride = inputs[0]->get_stride();
    Shape out_stride = outputs[0]->get_stride();

    int dim_a[] = {input_num, input_channel,
                   input_height, input_width};

    int dim_b[] = {input_num, output_channel,
                   output_height, output_width};

    cudnn::setTensorNdDesc<InDataType>(&_input_descs,
                                       inputs[0]->dims(), dim_a, &in_stride[0]);

    cudnn::setTensorNdDesc<InDataType>(&_output_descs,
                                      outputs[0]->dims(), dim_b, &out_stride[0]);

    int pad_a[] = {param.conv_param.pad_h, param.conv_param.pad_w};
    int filter_stride_a[] = {param.conv_param.stride_h, param.conv_param.stride_w};
    int dilation_a[] = {param.conv_param.dilation_h, param.conv_param.dilation_w};

    cudnn::setConvolutionNdDesc<OpDataType >(&_conv_descs,
                                          inputs[0]->dims() - 2, pad_a,
                                          filter_stride_a, dilation_a);
    // set activation descriptor
    if(param.has_active && !_use_saber_act) {
        cudnn::set_activation_des<OpDataType>(&_active_descs, param.activation_param.active);
    }

    // true: use tensor core
    // false: disable tensor core
    cudnn::set_math_type<OpDataType>(&_conv_descs, _use_tensor_core);
    cudnn::set_group_count<OpDataType>(&_conv_descs, param.conv_param.group);

    // Get fastest implement of cudnn
    // set up algo and workspace size
    if (param.conv_param.group == inputs[0]->channel() && \
        inputs[0]->channel() == outputs[0]->channel()) {
        //CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        _fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    } else {
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(_handle, \
            _input_descs, _filter_desc, _conv_descs, _output_descs, \
            _preference, _workspace_limit_bytes, &_fwd_algo));
    }

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_handle,
        _input_descs, _filter_desc, _conv_descs, _output_descs,
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
SaberStatus VenderConv2DAct<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
    dispatch(const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                ConvActiveParam<OpTensor>& param) {
    const InDataType *in_data = (const InDataType*)inputs[0]->data();
    InDataType *out_data = (InDataType*)outputs[0]->mutable_data();

    const float *weight_data = (const float *) param.conv_param.weight()->data();
    if (_use_saber_act || param.has_active == false) {
        CUDNN_CHECK(cudnnConvolutionForward(_handle,
                                            cudnn::cudnnTypeWrapper<float>::kOne(),
                                            _input_descs, in_data,
                                            _filter_desc, weight_data,
                                            _conv_descs,  _fwd_algo, 
                                            _workspace, _workspace_fwd_sizes,
                                            cudnn::cudnnTypeWrapper<float>::kZero(),
                                            _output_descs, out_data
        ));

        if (param.conv_param.bias()->size() > 0) {

            // add up bias.
            const float * bias_data = (const float*)param.conv_param.bias()->data();
            CUDNN_CHECK(cudnnAddTensor(_handle,
                                       cudnn::cudnnTypeWrapper<float>::kOne(),
                                       _bias_desc, bias_data,
                                       cudnn::cudnnTypeWrapper<float>::kOne(),
                                       _output_descs, out_data));

        }
        if (_use_saber_act) {
            _saber_act.dispatch(inputs, outputs, param.activation_param);
        }
        return SaberSuccess;
    }

    if (param.conv_param.bias()->size() > 0) {
        const float * bias_data = (const float*)param.conv_param.bias()->data();

        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(_handle,
            cudnn::cudnnTypeWrapper<float>::kOne(),
            _input_descs, in_data,
            _filter_desc, weight_data,
            _conv_descs, _fwd_algo, _workspace, _workspace_fwd_sizes,
            cudnn::cudnnTypeWrapper<float>::kZero(),
            _output_descs, out_data,
            _bias_desc,  bias_data,
            _active_descs, _output_descs, out_data));

    } else {

        CUDNN_CHECK(cudnnConvolutionForward(_handle,
            cudnn::cudnnTypeWrapper<float>::kOne(),
            _input_descs, in_data,
            _filter_desc, weight_data,
            _conv_descs,  _fwd_algo, _workspace, _workspace_fwd_sizes,
            cudnn::cudnnTypeWrapper<float>::kZero(),
            _output_descs, out_data
        ));

        CUDNN_CHECK(cudnnActivationForward(_handle, _active_descs,
            cudnn::cudnnTypeWrapper<InDataType>::kOne(),
            _input_descs, out_data,
            cudnn::cudnnTypeWrapper<InDataType>::kZero(),
            _output_descs, out_data
        ));
    }
    return SaberSuccess;
}

#if 0
template <>
SaberStatus VenderConv2DAct<NV, AK_INT8, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
    create(const std::vector<DataTensor_in *>& inputs,
            std::vector<DataTensor_out *>& outputs,
            ConvActiveParam<OpTensor>& param, Context<NV>& ctx> &ctx) {

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
    int in_size = inputs[0]->valid_size();
    int out_size = outputs[0]->valid_size();

    // ====== int8 conv, the input channel must be a multiple of 4
    CHECK_EQ(input_channel % 4, 0);

    int kernel_h = param.conv_param.weight()->height();
    int kernel_w = param.conv_param.weight()->width();

    int filter_dim_a[] = {output_channel,
                          input_channel,
                          kernel_h, kernel_w};

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(_filter_desc, CUDNN_DATA_INT8x4,
                                           CUDNN_TENSOR_NCHW_VECT_C,
                                           param.conv_param.weight()->dims(), filter_dim_a));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(_input_descs,
                    CUDNN_TENSOR_NCHW_VECT_C,
                    CUDNN_DATA_INT8x4,
                    input_num, input_channel, input_height, input_width));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(_output_descs,
                    CUDNN_TENSOR_NCHW_VECT_C,
                    CUDNN_DATA_INT8x4,
                    input_num, output_channel, output_height, output_width));
    // =====================================================================

    // for int8
    // These part is used to describe origin data layout;
    Shape in_stride = inputs[0]->get_stride();
    Shape out_stride = outputs[0]->get_stride();

    int dim_a[] = {input_num, input_channel,
                   input_height, input_width};

    int dim_b[] = {input_num, output_channel,
                   output_height, output_width};

    cudnn::setTensorNdDesc<DataTensor_in>(&_input_nchw_descs,
                                       inputs[0]->dims(), dim_a, &in_stride[0]);

    cudnn::setTensorNdDesc<DataTensor_in>(&_output_nchw_descs,
                                      outputs[0]->dims(), dim_b, &out_stride[0]);
    // =======
    int pad_a[] = {param.conv_param.pad_h, param.conv_param.pad_w};
    int filter_stride_a[] = {param.conv_param.stride_h, param.conv_param.stride_w};
    int dilation_a[] = {param.conv_param.dilation_h, param.conv_param.dilation_w};

    cudnn::setConvolutionNdDesc<OpDataType >(&_conv_descs,
                                          inputs[0]->dims() - 2, pad_a,
                                          filter_stride_a, dilation_a);
    // set activation descriptor
    cudnn::set_activation_des<OpDataType>(&_active_descs, param.activation_param.active);

    // true: use tensor core
    // false: disable tensor core
    cudnn::set_group_count<OpDataType>(&_conv_descs, param.conv_param.group);

    // Get fastest implement of cudnn
    _fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_handle,
                    _input_descs, _filter_desc, _conv_descs, _output_descs,
                    _fwd_algo, &_workspace_fwd_sizes));

    if (_workspace_fwd_sizes > _workspaceSizeInBytes) {
        _workspaceSizeInBytes = _workspace_fwd_sizes;
        if (_workspaceData != NULL) {
            cudaFree(_workspaceData);
        }
        cudaMalloc(&_workspaceData, _workspaceSizeInBytes);
        _workspace = reinterpret_cast<char*>(_workspaceData);
    }

    if (param.conv_param.bias()->size() > 0) {
        int dim_bias[] = {1, output_channel, 1, 1};
        int stride_bias[] = {output_channel, 1, 1, 1};

        cudnn::setTensorNdDesc<float >(&_bias_desc,
                                         4, dim_bias, stride_bias);
    }

    if (x8_data_size < in_size){
        x8_data_size = in_size;
        if (x8_data != NULL) {
            CUDA_CHECK(cudaFree(x8_data));
        }
        CUDA_CHECK(cudaMalloc(&x8_data,
                              sizeof(char) * x8_data_size));
    }

    if (y8_data_size < out_size){
        y8_data_size = out_size;
        if (y8_data != NULL) {
            CUDA_CHECK(cudaFree(y8_data));
        }
        CUDA_CHECK(cudaMalloc(&y8_data, sizeof(char) * y8_data_size));
    }
    return SaberSuccess;
}

template <>
SaberStatus VenderConv2DAct<AK_INT8>::dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ConvActiveParam<OpTensor>& param) {

    const void *in_data = (const void*)inputs[0]->data();
    void *out_data = (void*)outputs[0]->mutable_data();

    // scale data for int8
    float scale = 1.f;
    float scale_1 = 1 / scale;

    // int8 tensor transoform
    CUDNN_CHECK(cudnnTransformTensor(_handle,
                                     &scale,
                                     _input_nchw_descs, in_data,
                                     cudnn::cudnnTypeWrapper<float>::kZero(),
                                     _input_descs, x8_data));

    const void *weight_data = (const void *) param.conv_param.weight()->data();

    if (param.conv_param.bias()->size() > 0) {
        const float * bias_data = (const float*)param.conv_param.bias()->data();

        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(_handle,
            cudnn::cudnnTypeWrapper<float>::kOne(),
            _input_descs, x8_data,
            _filter_desc, weight_data,
            _conv_descs, _fwd_algo, workspace, _workspace_fwd_sizes,
            cudnn::cudnnTypeWrapper<float>::kZero(),
            _output_descs, y8_data,
            _bias_desc, bias_data,
            _active_descs, _output_descs, y8_data));
    } else {
        CUDNN_CHECK(cudnnConvolutionForward(_handle,
            cudnn::cudnnTypeWrapper<float>::kOne(),
            _input_descs, x8_data,
            _filter_desc, weight_data,
            _conv_descs,  _fwd_algo, workspace, _workspace_fwd_sizes,
            cudnn::cudnnTypeWrapper<float>::kZero(),
            _output_descs, y8_data));

        CUDNN_CHECK(cudnnActivationForward(_handle, _active_descs,
            cudnn::cudnnTypeWrapper<float>::kOne(),
            _input_descs, y8_data,
            cudnn::cudnnTypeWrapper<float>::kZero(),
            _output_descs, y8_data));
    }
    // int8 tensor transoform
    CUDNN_CHECK(cudnnTransformTensor(_handle,
        &scale_1,
        _output_descs, y8_data,
        cudnn::cudnnTypeWrapper<float>::kZero(),
        _output_nchw_descs, out_data));

    return SaberSuccess;

}

template <>
SaberStatus VenderConv2DAct<AK_INT8, AK_INT8, AK_INT8, NCHW_C4>::\
    create(const std::vector<DataTensor_in *>& inputs,
            std::vector<DataTensor_out *>& outputs,
            ConvActiveParam<OpTensor>& param, Context<NV>& ctx> &ctx) {
    CHECK_EQ(inputs[0]->dims(), 5);
    CHECK_EQ(inputs[0]->shape()[4], 4);
    CHECK_EQ(outputs[0]->dims(), 5);
    CHECK_EQ(outputs[0]->shape()[4], 4);
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
    int in_size = inputs[0]->valid_size();
    int out_size = outputs[0]->valid_size();

    // ====== int8 conv, the input channel must be a multiple of 4
            CHECK_EQ(input_channel % 4, 0);
    int kernel_h = param.conv_param.weight()->height();
    int kernel_w = param.conv_param.weight()->width();

    int filter_dim_a[] = {output_channel,
                          input_channel,
                          kernel_h, kernel_w};

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(_filter_desc, CUDNN_DATA_INT8x4,
                                           CUDNN_TENSOR_NCHW_VECT_C,
                                           4, filter_dim_a));

//    Shape in_stride = inputs[0]->get_stride();
//    Shape out_stride = outputs[0]->get_stride();

//    std::cout<<"in_stride";
//    for (auto i : in_stride) {
//        std::cout<<", "<<i;
//    }
//    std::cout<<std::endl;

//    cudnnSetTensor4dDescriptorEx(_input_descs, CUDNN_DATA_INT8x4,
//                                 input_num, input_channel, input_height, input_width,
//                                 in_stride[0], in_stride[1], in_stride[2], in_stride[3]);
//
//    cudnnSetTensor4dDescriptorEx(_output_descs, CUDNN_DATA_INT8x4,
//                                 input_num, output_channel, output_height, output_width,
//                                 out_stride[0], out_stride[1], out_stride[2], out_stride[3]);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(_input_descs,
                                           CUDNN_TENSOR_NCHW_VECT_C,
                                           CUDNN_DATA_INT8x4,
                                           input_num, input_channel, input_height, input_width));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(_output_descs,
                                           CUDNN_TENSOR_NCHW_VECT_C,
                                           CUDNN_DATA_INT8x4,
                                           input_num, output_channel, output_height, output_width));

    int pad_a[] = {param.conv_param.pad_h, param.conv_param.pad_w};
    int filter_stride_a[] = {param.conv_param.stride_h, param.conv_param.stride_w};
    int dilation_a[] = {param.conv_param.dilation_h, param.conv_param.dilation_w};

    cudnn::setConvolutionNdDesc<OpDataType >(&_conv_descs,
                                          2, pad_a,
                                          filter_stride_a, dilation_a);
    // set activation descriptor
    cudnn::set_activation_des<OpDataType>(&_active_descs, param.activation_param.active);

    // true: use tensor core
    // false: disable tensor core
    cudnn::set_group_count<OpDataType>(&_conv_descs, param.conv_param.group);

    // Get fastest implement of cudnn
    _fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_handle,
                   _input_descs, _filter_desc, _conv_descs, _output_descs,
                                                        _fwd_algo, &_workspace_fwd_sizes));

    if (_workspace_fwd_sizes > _workspaceSizeInBytes) {
        _workspaceSizeInBytes = _workspace_fwd_sizes;
        if (_workspaceData != NULL) {
            cudaFree(_workspaceData);
        }
        cudaMalloc(&_workspaceData, _workspaceSizeInBytes);
        _workspace = reinterpret_cast<char*>(_workspaceData);
    }

    if (param.conv_param.bias()->size() > 0) {
        int dim_bias[] = {1, output_channel, 1, 1};
        int stride_bias[] = {output_channel, 1, 1, 1};

        cudnn::setTensorNdDesc<float >(&_bias_desc,
                                           4, dim_bias, stride_bias);
    }
    return SaberSuccess;
}

template <>
SaberStatus VenderConv2DAct<AK_INT8, AK_INT8, AK_INT8, NCHW_C4>::dispatch(
                const std::vector<DataTensor *> inputs,
                std::vector<OutDataTensor *> outputs,
                ConvActiveParam<VenderConv2DAct::OpTensor> &param) {

    const void *in_data = (const void*)inputs[0]->data();
    void *out_data = (void*)outputs[0]->mutable_data();
    const void *weight_data = (const void *) param.conv_param.weight()->data();

    if (param.conv_param.bias()->size() > 0) {
        const float * bias_data = (const float*)param.conv_param.bias()->data();

        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(_handle,
               cudnn::cudnnTypeWrapper<float>::kOne(),
               _input_descs, in_data,
               _filter_desc, weight_data,
               _conv_descs, _fwd_algo, workspace, _workspace_fwd_sizes,
               cudnn::cudnnTypeWrapper<float>::kZero(),
               _output_descs, out_data,
               _bias_desc, bias_data,
               _active_descs, _output_descs, out_data));
    } else {

        CUDNN_CHECK(cudnnConvolutionForward(_handle,
               cudnn::cudnnTypeWrapper<float>::kOne(),
               _input_descs, in_data,
               _filter_desc, weight_data,
               _conv_descs,  _fwd_algo, workspace, _workspace_fwd_sizes,
               cudnn::cudnnTypeWrapper<float>::kZero(),
               _output_descs, out_data));

        CUDNN_CHECK(cudnnActivationForward(_handle, _active_descs,
               cudnn::cudnnTypeWrapper<float>::kOne(),
               _input_descs, out_data,
               cudnn::cudnnTypeWrapper<float>::kZero(),
               _output_descs, out_data));
    }

    return SaberSuccess;

}

template <>
SaberStatus VenderConv2DAct<AK_INT8, AK_INT8, AK_FLOAT, NCHW_C4, NCHW>::\
    create(const std::vector<DataTensor_in *>& inputs,
        std::vector<DataTensor_out *>& outputs,
        ConvActiveParam<OpTensor>& param, Context<NV>& ctx> &ctx) {

    CHECK_EQ(inputs[0]->dims(), 5);
    CHECK_EQ(inputs[0]->shape()[4], 4);
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
    int in_size = inputs[0]->valid_size();
    int out_size = outputs[0]->valid_size();

    // ====== int8 conv, the input channel must be a multiple of 4
            CHECK_EQ(input_channel % 4, 0);
    int kernel_h = param.conv_param.weight()->height();
    int kernel_w = param.conv_param.weight()->width();

    int filter_dim_a[] = {output_channel,
                          input_channel,
                          kernel_h, kernel_w};

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(_filter_desc, CUDNN_DATA_INT8x4,
               CUDNN_TENSOR_NCHW_VECT_C,
               4, filter_dim_a));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(_input_descs,
               CUDNN_TENSOR_NCHW_VECT_C,
               CUDNN_DATA_INT8x4,
               input_num, input_channel, input_height, input_width));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(_output_descs,
               CUDNN_TENSOR_NCHW,
               CUDNN_DATA_FLOAT,
               input_num, output_channel, output_height, output_width));

    int pad_a[] = {param.conv_param.pad_h, param.conv_param.pad_w};
    int filter_stride_a[] = {param.conv_param.stride_h, param.conv_param.stride_w};
    int dilation_a[] = {param.conv_param.dilation_h, param.conv_param.dilation_w};

    cudnn::setConvolutionNdDesc<OpDataType >(&_conv_descs,
               2, pad_a,
               filter_stride_a, dilation_a);
    // set activation descriptor
    cudnn::set_activation_des<OpDataType>(&_active_descs, param.activation_param.active);

    // true: use tensor core
    // false: disable tensor core
    cudnn::set_group_count<OpDataType>(&_conv_descs, param.conv_param.group);

    // Get fastest implement of cudnn
    _fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_handle,
               _input_descs, _filter_desc, _conv_descs, _output_descs,
               _fwd_algo, &_workspace_fwd_sizes));

    if (_workspace_fwd_sizes > _workspaceSizeInBytes) {
        _workspaceSizeInBytes = _workspace_fwd_sizes;
        if (_workspaceData != NULL) {
            cudaFree(_workspaceData);
        }
        cudaMalloc(&_workspaceData, _workspaceSizeInBytes);
        _workspace = reinterpret_cast<char*>(_workspaceData);
    }

    if (param.conv_param.bias()->size() > 0) {
        int dim_bias[] = {1, output_channel, 1, 1};
        int stride_bias[] = {output_channel, 1, 1, 1};

        cudnn::setTensorNdDesc<float >(&_bias_desc,
                                       4, dim_bias, stride_bias);
    }
    return SaberSuccess;
}

template <>
SaberStatus VenderConv2DAct<AK_INT8, AK_INT8, AK_FLOAT, NCHW_C4, NCHW>::\
    dispatch(const std::vector<DataTensor_in*>& inputs,
            std::vector<DataTensor_out*>& outputs,
            ConvActiveParam<OpTensor>& param) {

    const void *in_data = (const void*)inputs[0]->data();
    void *out_data = (void*)outputs[0]->mutable_data();
    const void *weight_data = (const void *) param.conv_param.weight()->data();

    if (param.has_active == false) {
        CUDNN_CHECK(cudnnConvolutionForward(_handle,
                                            cudnn::cudnnTypeWrapper<float>::kOne(),
                                            _input_descs, in_data,
                                            _filter_desc, weight_data,
                                            _conv_descs,  _fwd_algo, workspace, _workspace_fwd_sizes,
                                            cudnn::cudnnTypeWrapper<float>::kZero(),
                                            _output_descs, out_data
        ));

        if (param.conv_param.bias()->size() > 0) {

            // add up bias.
            const float * bias_data = (const float*)param.conv_param.bias()->data();
            CUDNN_CHECK(cudnnAddTensor(_handle,
                                       cudnn::cudnnTypeWrapper<float>::kOne(),
                                       _bias_desc, bias_data,
                                       cudnn::cudnnTypeWrapper<float>::kOne(),
                                       _output_descs, out_data));

        }
        return;
    }

    if (param.conv_param.bias()->size() > 0) {
        const float * bias_data = (const float*)param.conv_param.bias()->data();

        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(_handle,
               cudnn::cudnnTypeWrapper<float>::kOne(),
               _input_descs, in_data,
               _filter_desc, weight_data,
               _conv_descs, _fwd_algo, workspace, _workspace_fwd_sizes,
               cudnn::cudnnTypeWrapper<float>::kZero(),
               _output_descs, out_data,
               _bias_desc, bias_data,
               _active_descs, _output_descs, out_data));
    } else {

        CUDNN_CHECK(cudnnConvolutionForward(_handle,
               cudnn::cudnnTypeWrapper<float>::kOne(),
               _input_descs, in_data,
               _filter_desc, weight_data,
               _conv_descs,  _fwd_algo, workspace, _workspace_fwd_sizes,
               cudnn::cudnnTypeWrapper<float>::kZero(),
               _output_descs, out_data));

        CUDNN_CHECK(cudnnActivationForward(_handle, _active_descs,
               cudnn::cudnnTypeWrapper<float>::kOne(),
               _input_descs, out_data,
               cudnn::cudnnTypeWrapper<float>::kZero(),
               _output_descs, out_data));
    }

    return SaberSuccess;
}
#endif
}
}
