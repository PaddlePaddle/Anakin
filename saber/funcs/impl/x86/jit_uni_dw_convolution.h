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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_JIT_UNI_DW_CONVOLUTION_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_JIT_UNI_DW_CONVOLUTION_H

#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/funcs/impl/x86/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_dw_conv_kernel_f32.h"
#include "saber/saber_funcs_param.h"

namespace anakin{
namespace saber{

using namespace jit;
template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class JitUniDWConvolution : public ImplBase<Tensor<X86, inDtype, LayOutType_in>,
        Tensor<X86, outDtype, LayOutType_out>,
        Tensor<X86, OpDtype, LayOutType_op>,
        ConvActiveParam<Tensor<X86, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<X86, inDtype, LayOutType_in> inTensor;
    typedef Tensor<X86, outDtype, LayOutType_out> outTensor;
    typedef Tensor<X86, OpDtype, LayOutType_op> opTensor;
    typedef typename inTensor::Dtype dtype;

    JitUniDWConvolution()
        : kernel_(NULL)
    {}
    ~JitUniDWConvolution() {
        if (kernel_ != NULL) {
            delete kernel_;
        }
    }

    virtual SaberStatus init(const std::vector<inTensor*>& inputs,
                             std::vector<outTensor*>& outputs,
                             ConvActiveParam<opTensor> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<inTensor*>& inputs,
                               std::vector<outTensor*>& outputs,
                               ConvActiveParam<opTensor> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<inTensor*>& inputs,
                                 std::vector<outTensor*>& outputs,
                                 ConvActiveParam<opTensor> &param) override;

private:
    jit_conv_conf_t conf;
    jit_uni_dw_conv_kernel_f32<avx512_common> *kernel_;
    std::shared_ptr<opTensor> weights_internal;
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_JIT_UNI_DW_CONVOLUTION_H
