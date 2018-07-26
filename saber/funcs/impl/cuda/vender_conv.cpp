
#include "saber/funcs/impl/cuda/vender_conv.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/funcs/calibrate.h"

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

    if (param.bias()->size() > 0) {
        cudnn::createTensorDesc<OpDataType>(&_bias_desc);
    }
    cudnnCreateTensorDescriptor(&_input_nchw_descs);
    cudnnCreateTensorDescriptor(&_output_nchw_descs);

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

SaberStatus convert_weights(Tensor<X86>& out_tensor,
                            const Tensor<X86>& in_tensor,
                            Context<NV> ctx) {

    int input_channel = in_tensor.channel();
    int output_channel = out_tensor.valid_shape()[1];
    std::vector<float> vector_weight_scale;
    vector_weight_scale.resize(input_channel);

    int weight_inner_dim = in_tensor.channel()
                           * in_tensor.height()
                           * in_tensor.width();
    const float* in_weight_data = in_tensor.data();

    for (int c = 0; c < input_channel; ++c) {
        float max_val = -1.f;

        for (int i = 0; i < weight_inner_dim; ++i) {
            float read_data = fabs(in_weight_data[i]);
            max_val = (read_data > max_val) ? read_data : max_val;
        }

        vector_weight_scale[c] = max_val / 127.f;
        in_weight_data += weight_inner_dim;
        //                LOG(INFO)<<"max_val = "<<max_val<<" vector: "<<max_val / 127.f;
    }

    int o_num = out_tensor.num();
    int o_channel = output_channel;
    int o_height = out_tensor.height();
    int o_width = out_tensor.width();

    int out_n_stride = o_channel * o_height * o_width;
    int out_c_stride = o_height * o_width;
    int out_h_stride = o_width;

    Shape in_stride = in_tensor.get_stride();
    in_weight_data = in_tensor.data();
    char* out_weight_data = out_tensor.mutable_data();

    for (int idx = 0; idx < o_num * o_channel * o_height * o_width; ++idx) {

        int n = (idx / (out_n_stride)) % o_num;
        int in_offset = ((idx / (out_n_stride)) % o_num) * in_stride[0]
                        + ((idx / (out_c_stride)) % o_channel) * (in_stride[1] * 4)
                        + ((idx / (out_h_stride)) % o_height) * in_stride[2]
                        + (idx % o_width) * in_stride[3];

        int out_offset = ((idx / (out_n_stride)) % o_num) * out_n_stride
                         + ((idx / (out_c_stride)) % o_channel) * out_c_stride
                         + ((idx / (out_h_stride)) % o_height) * out_h_stride
                         + (idx % o_width);

        out_weight_data[out_offset * 4 + 0] = (char)(round(
                in_weight_data[in_offset + 0 * in_stride[1]] / vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 1] = (char)(round(
                in_weight_data[in_offset + 1 * in_stride[1]] / vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 2] = (char)(round(
                in_weight_data[in_offset + 2 * in_stride[1]] / vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 3] = (char)(round(
                in_weight_data[in_offset + 3 * in_stride[1]] / vector_weight_scale[n]));
    }
    out_tensor.set_scale(vector_weight_scale);
//    for (auto i : vector_weight_scale) {
//        LOG(INFO) << i;
//    }
    return SaberSuccess;
}

SaberStatus convert_bias(Tensor<X86>& out_tensor,
                         const Tensor<X86>& in_tensor,
                         float in_scale, std::vector<float> vector_weight_scale,
                         Context<NV> ctx) {
    unsigned long weight_size = vector_weight_scale.size();
    unsigned long bias_size = in_tensor.size();
    CHECK_GT(in_scale, 0);
    CHECK_GT(weight_size, 0);
    CHECK_EQ(bias_size, weight_size);

    const float* in_data = in_tensor.data();
    float* out_data = out_tensor.mutable_data();

    for (int i = 0; i < bias_size; ++i) {
        out_data[i] = in_data[i] / in_scale / vector_weight_scale[i];
    }

    return SaberSuccess;
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

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(_filter_desc, CUDNN_DATA_INT8x4,
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
                                             2, pad_a,
                                             filter_stride_a, dilation_a);

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
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               1, output_channel, 1, 1));
    }

    return SaberSuccess;
}

template <>
SaberStatus VenderConv2D<NV, AK_INT8>::\
    init(const std::vector<Tensor<NV> *>& inputs,
         std::vector<Tensor<NV> *>& outputs,
         ConvParam<NV>& param, Context<NV>& ctx) {

    bool use_int8 = true;
    use_int8 &= ((inputs[0]->channel() % 4) == 0);
    use_int8 &= ((outputs[0]->channel() % 4) == 0);

    if (!use_int8) {
        return SaberInvalidValue;
    } else {
        // prepare int8 memory
        Tensor<X86> weights_fp32_host;
        Tensor<X86> weights_int8_host;
        weights_fp32_host.re_alloc(param.weight()->valid_shape(), AK_FLOAT);
        weights_int8_host.re_alloc(param.weight()->valid_shape(), AK_INT8);
        int8_weights.re_alloc(param.weight()->valid_shape(), AK_INT8);
        weights_int8_host.set_layout(Layout_NCHW_C4);
        int8_weights.set_layout(Layout_NCHW_C4);
        weights_fp32_host.copy_from(*param.weight());
        convert_weights(weights_int8_host, weights_fp32_host, ctx);
        int8_weights.copy_from(weights_int8_host);
        int8_weights.set_scale(weights_int8_host.get_scale());

        cudaMalloc(&weights_scale, sizeof(float) * int8_weights.get_scale().size());
        cudaMemcpy(weights_scale, &(int8_weights.get_scale()[0]), sizeof(float) * int8_weights.get_scale().size(),
                   cudaMemcpyHostToDevice);
    }
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

    if (param.bias()->size() > 0) {
        cudnn::createTensorDesc<OpDataType>(&_bias_desc);
        if (use_int8) {
            float in_scale;
            if (inputs[0]->get_scale().size() == 1) {
                in_scale = inputs[0]->get_scale()[0];
            } else {
                LOG(FATAL) << "scale now support static calibrate only!!";
            }
            Tensor<X86> bias_fp32_host;
            Tensor<X86> bias_int32_host;
            bias_fp32_host.re_alloc(param.bias()->valid_shape(), AK_FLOAT);
            bias_int32_host.re_alloc(param.bias()->valid_shape(), AK_FLOAT);
            int32_bias.re_alloc(param.bias()->valid_shape(), AK_FLOAT);
            bias_fp32_host.copy_from(*param.bias());
            convert_bias(bias_int32_host, bias_fp32_host, in_scale, int8_weights.get_scale(), ctx);
            int32_bias.copy_from(bias_int32_host);
        }
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

    const void* in_data;
    void* out_data;
    float in_scale = 0.f;

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        if (inputs[0]->get_scale().size() == 1) {
            in_scale = inputs[0]->get_scale()[0];
        } else {
            LOG(FATAL) << "scale now support static calibrate only!!";
        }
        int8_input.re_alloc(inputs[0]->valid_shape(), AK_INT8);
        int8_input.set_layout(Layout_NCHW_C4);
        conv_calibrate_fp32_int8(int8_input, *inputs[0], in_scale, *(this->_ctx));
        in_data = (const void *)int8_input.data();
    } else {
        in_data = (const void*)inputs[0]->data();
    }

    out_data = (void*)outputs[0]->mutable_data();
    const void* weight_data = (const void*) int8_weights.data();

    CUDNN_CHECK(cudnnConvolutionForward(_handle,
                                        cudnn::cudnnTypeWrapper<float>::kOne(),
                                        _input_descs, in_data,
                                        _filter_desc, weight_data,
                                        _conv_descs, _fwd_algo, _workspace, _workspace_fwd_sizes,
                                        cudnn::cudnnTypeWrapper<float>::kZero(),
                                        _output_descs, out_data));
    if (param.bias()-> size() > 0) {
        // add up bias.
        const void* bias_data = (const void*)int32_bias.data();
        CUDNN_CHECK(cudnnAddTensor(_handle,
                                   cudnn::cudnnTypeWrapper<float>::kOne(),
                                   _bias_desc, bias_data,
                                   cudnn::cudnnTypeWrapper<float>::kOne(),
                                   _output_descs, out_data));
    }

    if (outputs[0]->get_dtype() == AK_FLOAT) {
        conv_calibrate_int32_fp32(
                *outputs[0], *outputs[0], in_scale, weights_scale, *_ctx);
    } else if (outputs[0]->get_dtype() == AK_INT8) {
        LOG(FATAL) << "not support output int8 now!!!";
    }
    return SaberSuccess;
};

template class VenderConv2D<NV, AK_FLOAT>;
template class VenderConv2D<NV, AK_INT8>;
}
}
