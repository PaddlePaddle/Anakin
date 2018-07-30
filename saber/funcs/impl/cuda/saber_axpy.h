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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_AXPY_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_AXPY_H

#include "saber/funcs/impl/impl_axpy.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberAxpy<NV, OpDtype> : \
    public ImplBase<
       NV, OpDtype,
        AxpyParam<NV>> 
{
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberAxpy()
    {}

    ~SaberAxpy() {

    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                        std::vector<Tensor<NV> *>& outputs,
                        AxpyParam<NV>& param, 
                        Context<NV> &ctx) {
        this->_ctx = &ctx;
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                        std::vector<Tensor<NV> *>& outputs,
                        AxpyParam<NV>& param, 
                        Context<NV>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                        std::vector<Tensor<NV> *>& outputs,
                        AxpyParam<NV>& param);

};

//template class SaberAxpy<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}

}

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_AXPY_H
