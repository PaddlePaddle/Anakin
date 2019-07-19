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
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_dwconv_kernel_f32.h"
#include "saber/saber_funcs_param.h"

namespace anakin {
namespace saber {

using namespace jit;

template<DataType OpDtype = AK_FLOAT>
class JitUniDWConv : public ImplBase<
        X86, OpDtype, ConvEltwiseParam <X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    JitUniDWConv()
            : kernel(nullptr)
    {}
    ~JitUniDWConv() {
        if (kernel != nullptr) {
            delete kernel;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             ConvEltwiseParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               ConvEltwiseParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 ConvEltwiseParam<X86> &param) override;

private:
    jit_conv_conf_t conf;
    jit_uni_dwconv_kernel_f32 *kernel = nullptr;
    std::shared_ptr<Tensor<X86> > weights_internal;
    SaberStatus check_conf(const std::vector<Tensor<X86>*>& inputs,
                           std::vector<Tensor<X86>*>& outputs,
                           ConvEltwiseParam<X86> &param);
};


} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_JIT_UNI_DW_CONVOLUTION_H
