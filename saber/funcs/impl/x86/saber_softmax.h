/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SOFTMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SOFTMAX_H

#include "saber/funcs/impl/impl_softmax.h"

namespace anakin{
namespace saber {

template <DataType OpDtype>
class SaberSoftmax<X86, OpDtype> : 
    public ImplBase<X86, OpDtype, SoftmaxParam<X86>> 
{
public:
    typedef Tensor<X86> DataTensor_in;
    typedef Tensor<X86> DataTensor_out;
    typedef Tensor<X86> OpTensor;

    SaberSoftmax()
    {}

    ~SaberSoftmax() {
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             SoftmaxParam<X86>& param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               SoftmaxParam<X86>& param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 SoftmaxParam<X86> &param) override;

private:
    void _max(int n, const float *x, float *max_data);
    void _sub(int n, float alpha, const float *x, float *y);
    void _exp(int n, const float *a, float *r);
    void _sum(int n, const float *x, float *sum_data);
    void _scal(int n, float alpha, float *x);
};

}
}

#endif