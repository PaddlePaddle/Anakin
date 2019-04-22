/* Copyright (c) 2018 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CORE_X8S8S32X_CONV_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CORE_X8S8S32X_CONV_H

#include "anakin_config.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_macro.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_x8s8s32x_conv_kernel.h"

namespace anakin {
namespace saber {

using namespace jit;

class JitAvx512X8S8S32XConv :
    public ImplBase<
        X86,
        AK_INT8,
        ConvEltwiseParam<X86> > {
public:
    typedef typename DataTrait<X86, AK_INT8>::Dtype OpDataType;

    JitAvx512X8S8S32XConv()
        : kernel_(nullptr), weights_internal_(nullptr), weights_padding_(nullptr),
        bias_internal_(nullptr), ws_(nullptr), ws_per_thread_(0),
        local_scales_(nullptr), compensation_(nullptr) {
    }

    ~JitAvx512X8S8S32XConv() {
        if (kernel_ != nullptr) {
            delete kernel_;
            kernel_ = nullptr;
        }

        if (bias_internal_ != nullptr) {
            delete bias_internal_;
            bias_internal_ = nullptr;
        }

        if (weights_internal_ != nullptr) {
            delete weights_internal_;
            weights_internal_ = nullptr;
        }

        if (weights_padding_ != nullptr) {
            delete weights_padding_;
            weights_padding_ = nullptr;
        }

        if (ws_ != nullptr) {
            delete ws_;
            ws_ = nullptr;
        }

        if (local_scales_ != nullptr) {
            delete local_scales_;
            local_scales_ = nullptr;
        }

        if (compensation_ != nullptr) {
            delete compensation_;
            compensation_ = nullptr;
        }

        std::vector<float>().swap(scale_);
    }

    virtual SaberStatus init(const std::vector<Tensor<X86>*> &inputs,
                             std::vector<Tensor<X86>*> &outputs,
                             ConvEltwiseParam<X86> &param,
                             Context<X86> &ctx);

    virtual SaberStatus create(const std::vector<Tensor<X86>*> &inputs,
                               std::vector<Tensor<X86>*> &outputs,
                               ConvEltwiseParam<X86> &param,
                               Context<X86> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*> &inputs,
                                 std::vector<Tensor<X86>*> &outputs,
                                 ConvEltwiseParam<X86> &param);


private:
    jit_avx512_core_x8s8s32x_fwd_kernel *kernel_;
    Tensor<X86>                         _weights_scale;
    Tensor<X86>                         _bias_scale;
    Tensor<X86>                         *weights_internal_;
    Tensor<X86>                         *weights_padding_;
    Tensor<X86>                         *bias_internal_;
    int                                 *ws_;
    size_t                              ws_per_thread_;
    float                               *local_scales_;
    int32_t                             *compensation_;

    float _sum_scale{0.f};

    // quantization scale(s)
    std::vector<float> scale_;

    virtual SaberStatus init_conf(jit_conv_conf_t &jcp,
                                  const std::vector<Tensor<X86>*> &inputs,
                                  std::vector<Tensor<X86>*> &outputs,
                                  ConvEltwiseParam<X86> &param);

    virtual SaberStatus check_conf(const jit_conv_conf_t &jcp,
                                   const std::vector<Tensor<X86>*> &inputs,
                                   std::vector<Tensor<X86>*> &outputs,
                                   ConvEltwiseParam<X86> &param);
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_X8S8S32X_CONV_H
