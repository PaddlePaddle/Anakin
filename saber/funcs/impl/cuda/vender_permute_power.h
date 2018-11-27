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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_PERMUTE_POWER_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_PERMUTE_POWER_H

#include "saber/funcs/impl/impl_permute_power.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class VenderPermutePower<NV, OpDtype>:\
    public ImplBase<
            NV, OpDtype,
            PermutePowerParam<NV> > {

public:

    VenderPermutePower()
            : _handle(NULL)
            , _input_descs(NULL)
            , _output_descs(NULL)
    {}

    ~VenderPermutePower() {

        if (_input_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_descs));
        }
        if (_output_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_descs));
        }
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             PermutePowerParam<NV> &param, \
                             Context<NV> &ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>&inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               PermutePowerParam<NV> &param, Context<NV> &ctx);
    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 PermutePowerParam<NV> &param);
private:
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _input_descs;
    cudnnTensorDescriptor_t _output_descs;
    const bool _use_tensor_core = true;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_PERMUTE_POWER_H
