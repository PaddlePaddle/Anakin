/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_DECONV_ACT_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_DECONV_ACT_H

#include "saber/funcs/impl/impl_deconv_act.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"

namespace anakin{

namespace saber{

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderDeconv2DAct<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        ConvActiveParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderDeconv2DAct()
            : _handle(NULL)
            , _workspaceData(NULL)
            , _workspace(NULL)
            , _conv_descs(NULL)
            , _input_descs(NULL)
            , _output_descs(NULL)
            , _filter_desc(NULL)
            , _bias_desc(NULL)
            , _workspace_bwd_sizes(0)
            , _workspaceSizeInBytes(0)
            , _bwd_algo(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
            , _active_descs(NULL)
    {}

    ~VenderDeconv2DAct() {

        if (_conv_descs) {
            CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(_conv_descs));
        }
        if (_input_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_descs));
        }
        if (_output_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_descs));
        }
        if (_filter_desc) {
            CUDNN_CHECK(cudnnDestroyFilterDescriptor(_filter_desc));
        }
        if (_bias_desc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_bias_desc));
        }
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
        if (_workspaceData != NULL) {
            CUDA_CHECK(cudaFree(_workspaceData));
        }
        if (_active_descs != NULL) {
            CUDNN_CHECK(cudnnDestroyActivationDescriptor(_active_descs));
        }
    }

    /**
     * [Create description] Init all cudnn resource here
     * @AuthorHTL
     * @DateTime  2018-02-01T16:13:06+0800
     * @param     inputs                    [description]
     * @param     outputs                   [description]
     * @param     conv_param                [conv parameters]
     */
    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActiveParam<OpTensor>& param, Context<NV>& ctx) {

        // ---- init cudnn resources ----

        _workspaceSizeInBytes = 0;
        _workspaceData = NULL;

        _workspace_bwd_sizes = 0;

        this->_ctx = ctx;
        // ---- get cuda resources ----

        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();

        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

        _workspace = NULL;

        int in_channels = inputs[0]->channel();

        // ---- create cudnn Descs ----
        cudnn::createFilterDesc<OpDataType>(&_filter_desc);

        cudnn::createTensorDesc<InDataType>(&_input_descs);
        cudnn::createTensorDesc<OutDataType>(&_output_descs);
        cudnn::createConvolutionDesc<OpDataType>(&_conv_descs);

        cudnn::create_activation_des<OpDataType>(&_active_descs);
        cudnn::createTensorDesc<OpDataType>(&_bias_desc);

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActiveParam<OpTensor>& param, Context<NV>& ctx) {

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

        //update_weights(param);

        int input_num = inputs[0]->num();
        int input_channel = inputs[0]->channel();
        int input_height = inputs[0]->height();
        int input_width = inputs[0]->width();

        int output_channel = outputs[0]->channel();
        int output_height = outputs[0]->height();
        int output_width = outputs[0]->width();

        int kernel_h = param.conv_param.weight()->height();
        int kernel_w = param.conv_param.weight()->width();

        int filter_dim_a[] = {input_channel, output_channel / param.conv_param.group, \
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

        cudnn::setTensorNdDesc<InDataType >(&_input_descs,
                                           inputs[0]->dims(), dim_a, &in_stride[0]);

        cudnn::setTensorNdDesc<OutDataType>(&_output_descs,
                                          outputs[0]->dims(), dim_b, &out_stride[0]);

        int pad_a[] = {param.conv_param.pad_h, param.conv_param.pad_w};
        int stride_a[] = {param.conv_param.stride_h, param.conv_param.stride_w};
        int dilation_a[] = {param.conv_param.dilation_h, param.conv_param.dilation_w};

        //cudnn::setConvolutionNdDesc<OpDataType >(&_conv_descs, \
                                              inputs[0]->dims() - 2, pad_a, \
                                              stride_a, dilation_a);
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(_conv_descs, \
                                                    pad_a[0], pad_a[1], \
                                                    stride_a[0], stride_a[1], \
                                                    dilation_a[0], dilation_a[1], \
                                                    CUDNN_CROSS_CORRELATION, \
                                                    cudnn::cudnnOpWrapper<OpDataType>::type));

        cudnn::set_activation_des<OpDataType>(&_active_descs, param.activation_param.active);

        // true: use tensor core
        // false: disable tensor core
        cudnn::set_math_type<OpDataType>(&_conv_descs, _use_tensor_core);
        cudnn::set_group_count<OpDataType>(&_conv_descs, param.conv_param.group);

        // Get fastest implement of cudnn
        // set up algo and workspace size
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(_handle, \
                         _filter_desc, _input_descs, _conv_descs, _output_descs, \
                        _preference, _workspace_limit_bytes, &_bwd_algo));

        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                _handle,
                _filter_desc, _input_descs, _conv_descs, _output_descs,
                _bwd_algo, &_workspace_bwd_sizes));

        if (_workspace_bwd_sizes > _workspaceSizeInBytes) {
            _workspaceSizeInBytes = _workspace_bwd_sizes;
            if (_workspaceData != NULL) {
                CUDA_CHECK(cudaFree(_workspaceData));
            }
            CUDA_CHECK(cudaMalloc(&_workspaceData, _workspaceSizeInBytes));
            _workspace = reinterpret_cast<char*>(_workspaceData);
        }

        if (param.conv_param.bias()->size() > 0){
            int dim_bias[] = {1, output_channel,1, 1};
            int stride_bias[] = {output_channel, 1, 1, 1};
            cudnn::setTensorNdDesc<OpDataType>(&_bias_desc,
                                            4, dim_bias, stride_bias);
        }
        return SaberSuccess;

    }
    void update_weights(ConvActiveParam<OpTensor> &param){};

    virtual SaberStatus dispatch(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActiveParam<OpTensor>& param) {
        const float* input_data = inputs[0]->data();
        float* output_data = outputs[0]->mutable_data();
        const float* weight_data = (const float*) param.conv_param.weight()->data();

        CUDNN_CHECK(cudnnConvolutionBackwardData(
                _handle,
                cudnn::cudnnTypeWrapper<float>::kOne(),
                _filter_desc, weight_data,
                _input_descs, input_data,
                _conv_descs,  _bwd_algo, _workspace, _workspace_bwd_sizes,
                cudnn::cudnnTypeWrapper<float>::kZero(),
                _output_descs, output_data));

        if (param.conv_param.bias()->size() > 0) {
            const float* bias_data;
            bias_data = (const float*)param.conv_param.bias()->data();
            CUDNN_CHECK(cudnnAddTensor(
                    _handle,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _bias_desc,
                    bias_data,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _output_descs,
                    output_data));
        }

        if (_active_descs) {
            CUDNN_CHECK(cudnnActivationForward(
                    _handle,
                    _active_descs,
                    cudnn::cudnnTypeWrapper<float>::kOne(),
                    _output_descs,
                    output_data,
                    cudnn::cudnnTypeWrapper<float>::kZero(),
                    _output_descs,
                    output_data));
        }

        return SaberSuccess;
    }
private:
    cudnnHandle_t _handle;
    cudnnConvolutionBwdDataAlgo_t _bwd_algo;

    cudnnTensorDescriptor_t _input_descs;
    cudnnTensorDescriptor_t _output_descs;
    cudnnTensorDescriptor_t _bias_desc;

    cudnnFilterDescriptor_t _filter_desc;

    cudnnConvolutionDescriptor_t _conv_descs;

    cudnnActivationDescriptor_t _active_descs;

    size_t _workspace_bwd_sizes;
    size_t _workspaceSizeInBytes;  // size of underlying storage

    void *_workspaceData;  // underlying storage
    void *_workspace;  // aliases into _workspaceData
    const bool _use_tensor_core = true;
    const size_t _workspace_limit_bytes = 64 * 1024 * 1024;
    const cudnnConvolutionBwdDataPreference_t _preference = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
};
template class VenderDeconv2DAct<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
}

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_DECONV_ACT_H