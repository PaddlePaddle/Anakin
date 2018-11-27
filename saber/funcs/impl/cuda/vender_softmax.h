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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_SOFTMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_SOFTMAX_H

#include "saber/funcs/impl/impl_softmax.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/saber_types.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class VenderSoftmax<NV, OpDtype>:
    public ImplBase<NV, OpDtype, SoftmaxParam<NV>> 
{
public:
    typedef Tensor<NV> DataTensor_in;
    typedef Tensor<NV> DataTensor_out;
    typedef Tensor<NV> OpTensor;
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    VenderSoftmax(){
        _handle = nullptr;
        _input_desc = nullptr;
        _output_desc = nullptr;
        _setup = false;
    }

    ~VenderSoftmax() {

        if (_setup){
            CUDNN_CHECK(cudnnDestroy(_handle));
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_desc));
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_desc));
        }
    }

    /**
     * \brief initial all cudnn resources here
     * @param inputs
     * @param outputs
     * @param param
     * @param ctx
     */
    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            SoftmaxParam<NV>& param, Context<NV>& ctx) ;

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            SoftmaxParam<NV>& param, Context<NV> &ctx);
    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          SoftmaxParam<NV> &param);

private:
    bool _setup{false};
    cudnnHandle_t             _handle;
    cudnnTensorDescriptor_t _input_desc;
    cudnnTensorDescriptor_t _output_desc;
};



} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_SOFTMAX_H
