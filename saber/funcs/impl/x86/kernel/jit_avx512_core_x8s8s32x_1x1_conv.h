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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CORE_X8S8S32X_1X1_CONV_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CORE_X8S8S32X_1X1_CONV_H

#include "anakin_config.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_macro.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_1x1_conv_utils.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_rtus_driver.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_x8s8s32x_1x1_conv_kernel.h"

#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {

using namespace jit;

class JitAvx512x8s8s32xConv1x1 : public ImplBase<
        X86,
        AK_INT8,
        ConvEltwiseParam<X86> > {
public:

    JitAvx512x8s8s32xConv1x1()
                : kernel_(nullptr), rtus_driver_(nullptr), scratch_(nullptr),
                  weights_internal_(nullptr), ws_per_thread_(0),
                  bias_internal_(nullptr), reduce_src(false), weights_padding_(nullptr),
                  local_scales_(nullptr), compensation_(nullptr) {
    }

    ~JitAvx512x8s8s32xConv1x1() {
        if (kernel_) {
            delete kernel_;
            kernel_ = nullptr;
        }
        if (rtus_driver_) {
            delete rtus_driver_;
            rtus_driver_ = nullptr;
        }
        if (scratch_) {
            zfree(scratch_);
            scratch_ = nullptr;
        }
        if (weights_internal_ != nullptr) {
            delete weights_internal_;
            weights_internal_ = nullptr;
        }

        if (weights_padding_ != nullptr) {
            delete weights_padding_;
            weights_padding_ = nullptr;
        }

        if (local_scales_ != nullptr) {
            delete local_scales_;
            local_scales_ = nullptr;
        }

        if (compensation_ != nullptr) {
            delete compensation_;
            compensation_ = nullptr;
        }

        delete bias_internal_;
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
    bool reduce_src;
    jit_avx512_core_x8s8s32x_conv1x1_kernel *kernel_;
    rtus_driver_t *rtus_driver_;
    size_t ws_per_thread_;
    uint8_t *scratch_;
    Tensor<X86>* weights_internal_;
    Tensor<X86>* weights_padding_;
    Tensor<X86>* bias_internal_;
    jit_1x1_conv_conf_t conf;
    conv_1x1_desc conv_d;
    float* local_scales_;
    int32_t* compensation_;
    // quantization scale(s)
    std::vector<float> scale_;

    float _sum_scale{0.f};
    Tensor<X86> _weights_scale;

    void prepare_rtus(const std::vector<Tensor<X86>*> &inputs, jit_1x1_conv_conf_t &jcp);

    SaberStatus check_conf(const std::vector<Tensor<X86>*> &inputs,
                           std::vector<Tensor<X86>*> &outputs,
                           ConvEltwiseParam<X86> &param);
};

} // namespace saber
} // namespace anakin

#endif // SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CORE_X8S8S32X_1X1_CONV_H
