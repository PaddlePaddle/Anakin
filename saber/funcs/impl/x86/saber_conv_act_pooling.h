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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONV_ACT_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONV_ACT_POOLING_H

#include "saber/funcs/impl/impl_conv_act_pooling.h"
#include "saber/funcs/impl/x86/jit_call_conf.h"

namespace anakin {
namespace saber {

using namespace jit;

template<DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberConv2DActPooling<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out> : public ImplBase<
        Tensor<X86, inDtype, LayOutType_in>,
        Tensor<X86, outDtype, LayOutType_out>,
        Tensor<X86, OpDtype, LayOutType_op>,
        ConvActivePoolingParam<Tensor<X86, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;
    typedef ConvActivePoolingParam<OpTensor> Param_t;
    typedef ImplBase<DataTensor_in, DataTensor_out, OpTensor, Param_t> Impl_t;
    typedef ConvActiveParam<OpTensor> Convact_param_t;
    typedef PoolingParam<OpTensor> Pooling_param_t;

    typedef ImplBase<DataTensor_in, DataTensor_out, OpTensor, Convact_param_t> Conv_impl_t;
    typedef ImplBase<DataTensor_in, DataTensor_out, OpTensor, Pooling_param_t> Pooling_impl_t;

    SaberConv2DActPooling()
            : c_impl(nullptr), p_impl(nullptr) {}

    ~SaberConv2DActPooling() {
        if (c_impl != nullptr) {
            delete c_impl;
        }
        if (p_impl != nullptr) {
            delete p_impl;
        }
        std::for_each(this->buf.begin(), this->buf.end(),
                      [&](DataTensor_out *t) {
                          delete t;
                          t = nullptr;
                      });
    }

    virtual SaberStatus init(const std::vector<DataTensor_in *> &inputs,
                             std::vector<DataTensor_out *> &outputs,
                             ConvActivePoolingParam<OpTensor> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in *> &inputs,
                               std::vector<DataTensor_out *> &outputs,
                               ConvActivePoolingParam<OpTensor> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in *> &inputs,
                                 std::vector<DataTensor_out *> &outputs,
                                 ConvActivePoolingParam<OpTensor> &param) override;

private:
    Conv_impl_t *c_impl = nullptr;
    Pooling_impl_t *p_impl = nullptr;
    std::vector<DataTensor_out *> buf;
};

}
}

#endif