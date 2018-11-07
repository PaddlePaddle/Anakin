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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_DECONV_H

#include "saber/funcs/impl/impl_deconv.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/cuda/saber_activation.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class VenderDeconv2D<NV, OpDtype> : \
    public ImplBase<NV, OpDtype, ConvParam<NV> > {
public:
    typedef Tensor<NV> OpTensor;
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;


    VenderDeconv2D()
        : _handle(NULL)
        , workspaceData(NULL)
        , workspace(NULL)
        , _conv_descs(NULL)
        , _input_descs(NULL)
        , _output_descs(NULL)
        , _filter_desc(NULL)
        , _bias_desc(NULL)
        , _workspace_bwd_sizes(0)
        , _workspaceSizeInBytes(0)
        , _bwd_algo(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
    {}

    ~VenderDeconv2D() {
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
        if (workspaceData != NULL) {
            CUDA_CHECK(cudaFree(workspaceData));
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
    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             ConvParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               ConvParam<NV>& param, Context<NV>& ctx);

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 ConvParam<NV>& param);

    SaberStatus trans_weights(Tensor<NV> &target_weights,
                              Tensor<NV> &target_bias,
                              int in_channel, int out_channel,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int dilation_h, int dilation_w,
                              int group) {
        return SaberUnImplError;
    }
private:
    bool _use_saber_act{false};
    SaberActivation<NV, OpDtype> _saber_act;
    cudnnHandle_t _handle;
    cudnnConvolutionBwdDataAlgo_t _bwd_algo;

    cudnnTensorDescriptor_t _input_descs;
    cudnnTensorDescriptor_t _output_descs;
    cudnnTensorDescriptor_t _bias_desc;
    cudnnFilterDescriptor_t _filter_desc;
    cudnnConvolutionDescriptor_t _conv_descs;

    size_t _workspace_bwd_sizes;
    size_t _workspaceSizeInBytes;  // size of underlying storage

    void* workspaceData;  // underlying storage
    void* workspace;  // aliases into workspaceData
    const bool _use_tensor_core = true;
    const size_t _workspace_limit_bytes = 4 * 1024 * 1024;
    const cudnnConvolutionBwdDataPreference_t _preference = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_DECONV_H