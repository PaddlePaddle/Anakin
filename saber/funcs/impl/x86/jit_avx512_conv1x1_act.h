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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CONV1X1_ACT_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CONV1X1_ACT_H

#include <typeinfo>
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/funcs/impl/x86/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_1x1_conv_utils.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_rtus_driver.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_conv1x1_act_kernel.h"

#include "x86_utils.h"

namespace anakin {
namespace saber {

namespace jit {
struct jit_avx512_common_1x1_conv_kernel;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class JitAvx512Conv1x1Act : public ImplBase<Tensor<X86, inDtype, LayOutType_in>,
        Tensor<X86, outDtype, LayOutType_out>,
        Tensor<X86, OpDtype, LayOutType_op>,
        ConvActiveParam<Tensor<X86, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<X86, inDtype, LayOutType_in> inTensor;
    typedef Tensor<X86, outDtype, LayOutType_out> outTensor;
    typedef Tensor<X86, OpDtype, LayOutType_op> opTensor;
    typedef typename inTensor::Dtype dtype;

    JitAvx512Conv1x1Act()
            : rtus_({}), jcp_({}),
              kernel_(nullptr), rtus_driver_(nullptr),
              scratch_(nullptr)
    {}
    
    ~JitAvx512Conv1x1Act() {
        if (kernel_) {
            delete kernel_;
        }
        if (rtus_driver_) {
            delete rtus_driver_;
        }
        if (scratch_) {
            zfree(scratch_);
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
    reduce_to_unit_stride_t rtus_;
    jit::jit_1x1_conv_conf_t jcp_;
    conv_1x1_desc conv_d_;
    jit::jit_avx512_common_1x1_conv_kernel *kernel_;
    jit::rtus_driver_t *rtus_driver_;
    size_t ws_per_thread_;
    float *scratch_; // TODO float
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CONV1X1_ACT_H
