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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_POOLING_H

#include "saber/funcs/impl/impl_pooling.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"

namespace anakin{

namespace saber {

template <DataType OpDtype >
class VenderPooling<NV, OpDtype>:\
 public ImplBase<
    NV,
    OpDtype,
    PoolingParam<NV>> {
    public:
        typedef Tensor<NV> DataTensor_in;
        typedef Tensor<NV> DataTensor_out;
        typedef Tensor<NV> OpTensor;
       
        VenderPooling() : _handle(NULL), _input_descs(NULL), _output_descs(NULL), _pooling_descs(NULL) {} 
        
        ~VenderPooling() {
            if (_handle != NULL) {
                CUDNN_CHECK(cudnnDestroy(_handle));
            }
            if (_input_descs) {
                CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_descs));
            }
            if (_output_descs) {
                CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_descs));
            }
            if(_pooling_descs) {
                cudnnDestroyPoolingDescriptor(_pooling_descs);
            }
        }
        
        virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 PoolingParam<NV> &pooling_param, Context<NV> &ctx);
        
        virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                                   std::vector<DataTensor_out*>& outputs,
                                   PoolingParam<NV> &pooling_param, Context<NV> &ctx);
        
        virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                     std::vector<DataTensor_out*>& outputs,
                                     PoolingParam<NV> &param);
        
    private:
        cudnnHandle_t _handle;
        cudnnTensorDescriptor_t _input_descs;
        cudnnTensorDescriptor_t _output_descs;
        cudnnPoolingDescriptor_t _pooling_descs;
        
};

} //namespace saber

} // namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_POOLING_H
