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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_NORMALIZE_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_NORMALIZE_H

#include "saber/funcs/impl/impl_normalize.h"

namespace anakin {

namespace saber {

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class SaberNormalize<X86, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>: \
    public ImplBase <
    Tensor<X86, inDtype, LayOutType_in>,
    Tensor<X86, outDtype, LayOutType_out>,
    Tensor<X86, OpDtype, LayOutType_op>,
    NormalizeParam<Tensor<X86, OpDtype, LayOutType_op> >> {

public:
    typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberNormalize() = default;
    ~SaberNormalize() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             NormalizeParam<OpTensor>& param,
                             Context<X86>& ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               NormalizeParam<OpTensor>& param,
                               Context<X86>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 NormalizeParam<OpTensor>& param);


private:


};
template class SaberNormalize<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_NORMALIZE_H