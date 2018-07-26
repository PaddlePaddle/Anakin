/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_CONV2D_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_CONV2D_H

#include "saber/funcs/impl/impl_conv.h"
#include <cudnn.h>

namespace anakin{

namespace saber{

template <DataType OpDtype>
class VenderConv2D<NV, OpDtype> : public ImplBase<
        NV, OpDtype, ConvParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    VenderConv2D()
            : _handle(NULL)
            , _workspaceData(NULL)
            , _workspace(NULL)
            , _conv_descs(NULL)
            , _input_descs(NULL)
            , _output_descs(NULL)
            , _filter_desc(NULL)
            , _workspace_fwd_sizes(0)
            , _workspaceSizeInBytes(0)
            , _fwd_algo((cudnnConvolutionFwdAlgo_t)0)
            , _input_nchw_descs(NULL)
            , _bias_desc(NULL)
            , _output_nchw_descs(NULL)
            , weights_scale(NULL)
    {}

    ~VenderConv2D() {

        if (_conv_descs) {
            CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(_conv_descs));
        }
        if (_input_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_descs));
        }
        if (_output_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_descs));
        }
        if (_bias_desc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_bias_desc));
        }
        if (_filter_desc) {
            CUDNN_CHECK(cudnnDestroyFilterDescriptor(_filter_desc));
        }
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
        if (_workspaceData != NULL) {
            cudaFree(_workspaceData);
        }
        if (_input_nchw_descs != NULL) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_nchw_descs));
        }
        if (_output_nchw_descs != NULL) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_nchw_descs));
        }
        cudaFree(weights_scale);
    }

    /**
     * [Create description] Init all cudnn resource here
     * @AuthorHTL
     * @DateTime  2018-02-01T16:13:06+0800
     * @param     inputs                    [description]
     * @param     outputs                   [description]
     * @param     param                [conv parameters]
     */
    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             ConvParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               ConvParam<NV>& param, Context<NV>& ctx);

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 ConvParam<NV>& param);

private:
    cudnnHandle_t _handle;
    cudnnConvolutionFwdAlgo_t _fwd_algo;
    cudnnTensorDescriptor_t _input_descs;
    cudnnTensorDescriptor_t _output_descs;
    cudnnTensorDescriptor_t _bias_desc;
    cudnnFilterDescriptor_t _filter_desc;
    cudnnConvolutionDescriptor_t _conv_descs;

    size_t _workspace_fwd_sizes;
    size_t _workspaceSizeInBytes;  // size of underlying storage
    void *_workspaceData;  // underlying storage
    void *_workspace;  // aliases into _workspaceData
    const bool _use_tensor_core = true;
    const size_t _workspace_limit_bytes = 4 * 1024 * 1024;
    const cudnnConvolutionFwdPreference_t _preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;

    // create transform descriptor
    cudnnTensorDescriptor_t _input_nchw_descs;
    cudnnTensorDescriptor_t _output_nchw_descs;

    float* weights_scale;
    Tensor<NV> int8_weights;
    Tensor<NV> int8_input;
    Tensor<NV> int8_output;
};


}

}
#endif //ANAKIN_SABER_FUNCS_CUDNN_CONV2D_H
