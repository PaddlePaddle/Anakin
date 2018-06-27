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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_CONV_ACT_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_CONV_ACT_POOLING_H

#include "saber/funcs/impl/impl_conv_act_pooling.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"   
#include <cudnn.h>

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderConv2DActPooling<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        ConvActivePoolingParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderConv2DActPooling()
            : _handle(NULL)
            , _workspaceData(NULL)
            , _workspace(NULL)
            , _conv_descs(NULL)
            , _input_descs(NULL)
            , _output_descs(NULL)
            , _filter_desc(NULL)
            , _inner_descs(NULL)
            , _bias_desc(NULL)
            , _pooling_descs(NULL)
            , _active_descs(NULL)
            , _workspace_fwd_sizes(0)
            , _workspaceSizeInBytes(0)
            , _fwd_algo((cudnnConvolutionFwdAlgo_t)0)
    {}
    ~VenderConv2DActPooling() {

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
        if (_handle) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
        if (_workspaceData) {
            CUDA_CHECK(cudaFree(_workspaceData));
        }
        if (_active_descs) {
            CUDNN_CHECK(cudnnDestroyActivationDescriptor(_active_descs));
        }
        if (_inner_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_inner_descs));
        }
        if (_bias_desc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_bias_desc));
        }
        if (_pooling_descs) {
            CUDNN_CHECK(cudnnDestroyPoolingDescriptor(_pooling_descs));
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
                            ConvActivePoolingParam<OpTensor>& param, Context<NV>& ctx) {
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

        cudnn::createTensorDesc<InDataType>(&_input_descs);
        cudnn::createTensorDesc<InDataType>(&_inner_descs);
        cudnn::createTensorDesc<OutDataType>(&_output_descs);
        cudnn::createConvolutionDesc<OpDataType>(&_conv_descs);
        if (param.has_activation) {
            cudnn::create_activation_des<InDataType>(&_active_descs);
        }
        if (param.has_pooling) {
            cudnn::create_pooling_des<InDataType>(&_pooling_descs);
        }
        if (param.conv_param.bias()->size() > 0) {
            cudnn::createTensorDesc<OpDataType>(&_bias_desc);
        }

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActivePoolingParam<OpTensor>& param, Context<NV>& ctx);
    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ConvActivePoolingParam<OpTensor>& param);
private:
    cudnnHandle_t _handle;
    cudnnConvolutionFwdAlgo_t _fwd_algo;

    cudnnTensorDescriptor_t _input_descs;
    cudnnTensorDescriptor_t _output_descs;
    cudnnTensorDescriptor_t _inner_descs;
    cudnnTensorDescriptor_t _bias_desc;

    cudnnFilterDescriptor_t _filter_desc;

    cudnnConvolutionDescriptor_t _conv_descs;
    cudnnPoolingDescriptor_t _pooling_descs;

    size_t _workspace_fwd_sizes;
    size_t _workspaceSizeInBytes;  // size of underlying storage

    void *_workspaceData;  // underlying storage
    void *_workspace;  // aliases into workspaceData

    const bool _use_tensor_core = true;
    const size_t _workspace_limit_bytes = 4 * 1024 * 1024;
    const cudnnConvolutionFwdPreference_t _preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;

    // activation descriptor
    cudnnActivationDescriptor_t _active_descs;

    Shape _inner_shape;
    DataTensor_out _inner_tensor;
};


}

}
#endif //ANAKIN_SABER_FUNCS_CUDNN_CONV2D_H
