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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CONV1X1_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CONV1X1_H

#include <typeinfo>
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_1x1_conv_utils.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_rtus_driver.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_conv1x1_kernel.h"

#include "x86_utils.h"

namespace anakin {
namespace saber {

using namespace jit;

template<DataType OpDtype = AK_FLOAT>
class JitAvx512Conv1x1 : public ImplBase<
        X86, OpDtype, ConvEltwiseParam <X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    JitAvx512Conv1x1()
            : conf(),
              kernel(nullptr), rtus_driver(nullptr),
              scratch(nullptr), weights_internal(nullptr), ws_per_thread(0), reduce_src(false)
    {}

    ~JitAvx512Conv1x1() {
        if (kernel) {
            delete kernel;
            kernel = nullptr;
        }
        if (rtus_driver) {
            delete rtus_driver;
            rtus_driver = nullptr;
        }
        if (scratch) {
            zfree(scratch);
            scratch = nullptr;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             ConvEltwiseParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               ConvEltwiseParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 ConvEltwiseParam<X86> &param) override;
private:
    conv_1x1_desc conv_d;
    jit_1x1_conv_conf_t conf;
    bool reduce_src = false;
    jit_avx512_common_1x1_conv_kernel *kernel = nullptr;
    rtus_driver_t *rtus_driver = nullptr;
    size_t ws_per_thread;
    OpDataType *scratch = nullptr;
    std::shared_ptr<Tensor<X86> > weights_internal = nullptr;

    void prepare_rtus();
    SaberStatus check_conf(const std::vector<Tensor<X86> *>& inputs,
                           std::vector<Tensor<X86>*>& outputs,
                           ConvEltwiseParam<X86> &param);
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CONV1X1_H
