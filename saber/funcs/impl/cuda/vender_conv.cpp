
#include "saber/funcs/impl/cuda/vender_conv.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/funcs/calibrate.h"
#include "saber/core/tensor_op.h"

namespace anakin {
namespace saber {

// FP32 part
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

    cudnn::setTensorNdDesc<float >(&_input_descs,
            inputs[0]->dims(), dim_a, &in_stride[0]);

    cudnn::setTensorNdDesc<float>(&_output_descs,
            outputs[0]->dims(), dim_b, &out_stride[0]);

    int pad_a[] = {param.pad_h, param.pad_w};
    int filter_stride_a[] = {param.stride_h, param.stride_w};
    int dilation_a[] = {param.dilation_h, param.dilation_w};

    cudnn::setConvolutionNdDesc<OpDataType >(&_conv_descs,
            inputs[0]->dims() - 2, pad_a, filter_stride_a, dilation_a);

    if (param.activation_param.has_active && !_with_saber_act) {
        cudnn::set_activation_des<OpDataType>(&_active_descs, param.activation_param.active);
    }
    if (_with_saber_act) {
        _saber_act->create(inputs, outputs, param.activation_param, ctx);
    }
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
SaberStatus VenderConv2D<NV, AK_FLOAT>::\
    init(const std::vector<Tensor<NV> *>& inputs,
           std::vector<Tensor<NV> *>& outputs,
           ConvParam<NV>& param, Context<NV>& ctx) {

    // ---- init cudnn resources ----
    _workspaceSizeInBytes = 0;
    _workspaceData = NULL;
    _workspace_fwd_sizes = 0;

    this->_ctx = &ctx;
    // ---- get cuda resources ----
    cudaStream_t cuda_stream;
    cuda_stream = ctx.get_compute_stream();
    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

    _workspace = NULL;
    int in_channels = inputs[0]->channel();
    // ---- create cudnn Descs ----
    cudnn::createFilterDesc<OpDataType>(&_filter_desc);
    cudnn::createTensorDesc<OpDataType>(&_input_descs);
    cudnn::createTensorDesc<OpDataType>(&_output_descs);
    cudnn::createConvolutionDesc<OpDataType>(&_conv_descs);

    if (param.activation_param.has_active) {
        if (param.activation_param.active == Active_relu
            && fabs(param.activation_param.negative_slope) < 1e-5) {
            cudnn::create_activation_des<OpDataType>(&_active_descs);
        } else {
            _with_saber_act = true;
        }
    }
    if (param.bias()->size() > 0) {
        cudnn::createTensorDesc<OpDataType>(&_bias_desc);
    }
    cudnnCreateTensorDescriptor(&_input_nchw_descs);
    cudnnCreateTensorDescriptor(&_output_nchw_descs);
    if (_with_saber_act) {
        _saber_act = new SaberActivation<NV, AK_FLOAT>;
        _saber_act->init(outputs, outputs, param.activation_param, ctx);
    }
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderConv2D<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ConvParam<NV>& param) {

    const float* in_data = (const float*)inputs[0]->data();
    float* out_data = (float*)outputs[0]->mutable_data();
    const float* weight_data = (const float*) param.weight()->data();

    if (param.activation_param.has_active && !_with_saber_act) {
        if (param.bias()->size() > 0) {
            const float * bias_data = (const float*)param.bias()->data();
            CUDNN_CHECK(cudnnConvolutionBiasActivationForward(_handle,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _input_descs, in_data,
                    _filter_desc, weight_data,
                    _conv_descs, _fwd_algo,
                    _workspace, _workspace_fwd_sizes,
                    &_beta, _output_descs,
                    out_data, _bias_desc, bias_data,
                    _active_descs, _output_descs, out_data));
        } else {
            CUDNN_CHECK(cudnnConvolutionForward(_handle,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _input_descs, in_data,
                    _filter_desc, weight_data,
                    _conv_descs,  _fwd_algo,
                    _workspace, _workspace_fwd_sizes,
                    &_beta, _output_descs, out_data));

            CUDNN_CHECK(cudnnActivationForward(_handle, _active_descs,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _output_descs, out_data,
                    &_beta, _output_descs, out_data));
        }
    } else {
        CUDNN_CHECK(cudnnConvolutionForward(_handle,
                cudnn::cudnnTypeWrapper<float>::kOne(),
                _input_descs, in_data,
                _filter_desc, weight_data,
                _conv_descs, _fwd_algo,
                _workspace, _workspace_fwd_sizes,
                &_beta, _output_descs, out_data));

        if (param.bias()->size() > 0) {
            // add up bias.
            const float *bias_data = (const float *) param.bias()->data();
            CUDNN_CHECK(cudnnAddTensor(_handle,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _bias_desc, bias_data,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _output_descs, out_data));
        }
    }
    if (_with_saber_act) {
        _saber_act->dispatch(outputs, outputs, param.activation_param);
    }
    return SaberSuccess;
}
template <>
SaberStatus VenderConv2D<NV, AK_FLOAT>::trans_weights(Tensor<NV> &target_weights,
        Tensor<NV> &target_bias,int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberUnImplError;
}

// INT8 part
template <>
SaberStatus VenderConv2D<NV, AK_INT8>::\
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
    int in_size = inputs[0]->valid_size();
    int out_size = outputs[0]->valid_size();

    // ====== int8 conv, the input channel must be a multiple of 4
    CHECK_EQ(input_channel % 4, 0);

    int kernel_h = param.weight()->height();
    int kernel_w = param.weight()->width();

    int filter_dim_a[] = {output_channel,
                          input_channel,
                          kernel_h, kernel_w};

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(_filter_desc,
            CUDNN_DATA_INT8x4,
            CUDNN_TENSOR_NCHW_VECT_C,
            4, filter_dim_a));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(_input_descs,
            CUDNN_TENSOR_NCHW_VECT_C,
            CUDNN_DATA_INT8x4,
            input_num, input_channel,
            input_height, input_width));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(_output_descs,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            input_num, output_channel,
            output_height, output_width));

    int pad_a[] = {param.pad_h, param.pad_w};
    int filter_stride_a[] = {param.stride_h, param.stride_w};
    int dilation_a[] = {param.dilation_h, param.dilation_w};

    cudnn::setConvolutionNdDesc<OpDataType >(&_conv_descs,
            2, pad_a, filter_stride_a, dilation_a);

    if(param.activation_param.has_active) {
        cudnn::set_activation_des<OpDataType>(&_active_descs,
                param.activation_param.active);
    }

    // true: use tensor core
    // false: disable tensor core
    cudnn::set_group_count<OpDataType>(&_conv_descs, param.group);

    // Get fastest implement of cudnn
    _fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
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
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(_bias_desc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, output_channel, 1, 1));
    }

    return SaberSuccess;
}

template <>
SaberStatus VenderConv2D<NV, AK_INT8>::trans_weights(Tensor<NV> &target_weights,
        Tensor<NV> &target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    if (target_weights.valid_size() == 0) {
        return SaberSuccess;
    }
    if (target_weights.get_dtype() == AK_INT8) {
        return SaberSuccess;
    }
    if (target_weights.channel() % 4 == 0 && target_weights.num() % 4 == 0) {
        // prepare int8 memory
        Tensor<NVHX86> weights_fp32_host;
        Tensor<NVHX86> weights_int8_host;
        weights_fp32_host.re_alloc(target_weights.valid_shape(), AK_FLOAT);
        weights_int8_host.re_alloc(target_weights.valid_shape(), AK_INT8);
        weights_int8_host.set_layout(Layout_NCHW_C4);
        weights_fp32_host.copy_from(target_weights);
        convert_weights_to_nchw_c4_host(weights_int8_host, weights_fp32_host, *_ctx);
        // Open this will be an inplace trans

        target_weights.set_dtype(AK_INT8);
        target_weights.re_alloc(target_weights.valid_shape(), AK_INT8);
        target_weights.set_layout(Layout_NCHW_C4);
        target_weights.copy_from(weights_int8_host);
        target_weights.set_scale(weights_int8_host.get_scale());
        if (target_bias.valid_size() > 0) {
            Tensor<NVHX86> bias_fp32_host;
            Tensor<NVHX86> bias_int32_host;
            bias_fp32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_int32_host.re_alloc(target_bias.valid_shape(), AK_FLOAT);
            bias_fp32_host.copy_from(target_bias);
            convert_bias_host(bias_int32_host, bias_fp32_host, _in_scale,
                    target_weights.get_scale(), *_ctx);
            target_bias.copy_from(bias_int32_host);
        }
    }
    return SaberSuccess;
}
template <>
SaberStatus VenderConv2D<NV, AK_INT8>::\
    init(const std::vector<Tensor<NV> *>& inputs,
         std::vector<Tensor<NV> *>& outputs,
         ConvParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;


    // ---- init cudnn resources ----
    _workspaceSizeInBytes = 0;
    _workspaceData = NULL;
    _workspace_fwd_sizes = 0;
    // ---- get cuda resources ----
    cudaStream_t cuda_stream;
    cuda_stream = ctx.get_compute_stream();

    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

    _workspace = NULL;
    int in_channels = inputs[0]->channel();

    // ---- create cudnn Descs ----
    cudnn::createFilterDesc<OpDataType>(&_filter_desc);
    cudnn::createTensorDesc<OpDataType>(&_input_descs);
    cudnn::createTensorDesc<OpDataType>(&_output_descs);
    cudnn::createConvolutionDesc<OpDataType>(&_conv_descs);
    if (param.activation_param.has_active) {
        cudnn::create_activation_des<OpDataType>(&_active_descs);
    }
    if (param.bias()->size() > 0) {
        cudnn::createTensorDesc<OpDataType>(&_bias_desc);
    }

    cudnnCreateTensorDescriptor(&_input_nchw_descs);
    cudnnCreateTensorDescriptor(&_output_nchw_descs);

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderConv2D<NV, AK_INT8>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ConvParam<NV>& param) {
//    LOG(INFO) << "conv int8 dispatch"
//                << " input tensor dtype: " << (inputs[0]->get_dtype() == AK_FLOAT ? "AK_FLOAT" : "AK_INT8")
//                << " output tensor dtype: " << (outputs[0]->get_dtype() == AK_FLOAT ? "AK_FLOAT" : "AK_INT8");
    const void* in_data = (const void*)inputs[0]->data();
    void* out_data = (void*)outputs[0]->mutable_data();
    const void* weight_data = (const void*) param.weight()->data();

    if (param.activation_param.has_active) {
        if (param.bias()->valid_size() > 0) {
            const void *bias_data = (const void *) param.bias()->data();
            CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
                    _handle, cudnn::cudnnTypeWrapper<float>::kOne(),
                    _input_descs, in_data, _filter_desc, weight_data,
                    _conv_descs, _fwd_algo, _workspace, _workspace_fwd_sizes,
                    cudnn::cudnnTypeWrapper<float>::kZero(),
                    _output_descs, out_data,
                    _bias_desc, bias_data,
                    _active_descs, _output_descs, out_data));
        } else {
            CUDNN_CHECK(cudnnConvolutionForward(_handle,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _input_descs, in_data,
                    _filter_desc, weight_data,
                    _conv_descs, _fwd_algo,
                    _workspace, _workspace_fwd_sizes,
                    cudnn::cudnnTypeWrapper<float>::kZero(),
                    _output_descs, out_data));

            CUDNN_CHECK(cudnnActivationForward(_handle, _active_descs,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _output_descs, out_data,
                    cudnn::cudnnTypeWrapper<float>::kZero(),
                    _output_descs, out_data));
        }
    } else {
        CUDNN_CHECK(cudnnConvolutionForward(_handle,
                cudnn::cudnnTypeWrapper<float>::kOne(),
                _input_descs, in_data,
                _filter_desc, weight_data,
                _conv_descs, _fwd_algo,
                _workspace, _workspace_fwd_sizes,
                cudnn::cudnnTypeWrapper<float>::kZero(),
                _output_descs, out_data));
        if (param.bias()->size() > 0) {
            // add up bias.
            const void *bias_data = (const void *) param.bias()->data();
            CUDNN_CHECK(cudnnAddTensor(_handle,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _bias_desc, bias_data,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _output_descs, out_data));
        }
    }
    return SaberSuccess;
}

template <>
SaberStatus VenderConv2D<NV, AK_HALF>::trans_weights(Tensor<NV> &target_weights,
        Tensor<NV> &target_bias, int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberUnImplError;
}

template class VenderConv2D<NV, AK_FLOAT>;
template class VenderConv2D<NV, AK_INT8>;
DEFINE_OP_TEMPLATE(VenderConv2D, ConvParam, NV, AK_HALF);
}
}
