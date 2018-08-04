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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_POOLING_H

#include <vector>

#include "anakin_config.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/core/context.h"
#include "saber/funcs/impl/impl_pooling.h"
namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPooling<NV, OpDtype>:\
    public ImplBase<
            NV,OpDtype,
            PoolingParam<NV>> {
        
    public:
        typedef Tensor<NV> DataTensor_in;
        typedef Tensor<NV> DataTensor_out;
        
        SaberPooling(){}
        
        ~SaberPooling() {}
        
        virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 PoolingParam<NV> &param,
                                 Context<NV> &ctx) override {
            
            return SaberUnImplError;
            
        }
        
        virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                                   std::vector<DataTensor_out*>& outputs,
                                   PoolingParam<NV> &param,
                                   Context<NV> &ctx) override {
            
            return SaberUnImplError;
            
        }
        
        //call cudnnConvolutionForward here
        virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                     std::vector<DataTensor_out*>& outputs,
                                     PoolingParam<NV> &param) {
            
            return SaberUnImplError;
        }
};

}

}


#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
