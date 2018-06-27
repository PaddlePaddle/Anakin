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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_MAT_MUL_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_MAT_MUL_H

#include "saber/funcs/impl/impl_mat_mul.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
namespace anakin{

namespace saber{

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
class SaberMatMul<NV, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>,
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        MatMulParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberMatMul() {}

    ~SaberMatMul() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             MatMulParam<OpTensor> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               MatMulParam<OpTensor> &param,
                               Context<NV> &ctx) {
        _kernel =saber_find_fast_sass_gemm(param._is_transpose_X, param._is_transpose_Y, param._m, param._n, param._k);
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 MatMulParam<OpTensor>  &param)
    {
        cudaStream_t stream = this->_ctx->get_compute_stream();
        const InDataType* X = inputs[0]->data();
        const InDataType* Y = inputs[1]->data();
        OutDataType* out = outputs[0]->mutable_data();

        //should add batch gemm here
        for (int b = 0; b < param._b; b++)
        {
            _kernel(param._m, param._n, param._k, 1.f,
                X + b * param._m * param._k,
                0.f, 
                Y + b * param._k * param._n,
                out + b * param._m * param._n, stream);
        }
        return SaberSuccess;
    }

private:

    std::function<void(const int, const int, const int,
                    const float, const float*, const float,
                    const float*, float*, cudaStream_t)> _kernel;
};

template class SaberMatMul<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_MAT_MUL_H