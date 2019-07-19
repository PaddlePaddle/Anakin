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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CONV_POOL_OPTIMIZED_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CONV_POOL_OPTIMIZED_H

#include "anakin_config.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_macro.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_u8s8s32x_conv_act_pool_kernel.h"

namespace anakin {
namespace saber {

using namespace jit;

class JitAvx512ConvPoolOptimized :
    public ImplBase<
        X86,
        AK_INT8,
        ConvPoolingParam<X86>> {
public:

    typedef ImplBase<X86, AK_INT8, PoolingParam<X86> > Impl_pool_t;
    JitAvx512ConvPoolOptimized()
        : conv_kernel_(nullptr), pool_impl_(nullptr),
        weights_internal_(nullptr), bias_internal_(nullptr)
    {}

    ~JitAvx512ConvPoolOptimized() {
        if (conv_kernel_ != nullptr) {
            delete conv_kernel_;
            conv_kernel_ = nullptr;
        }

        if (pool_impl_ != nullptr) {
            delete pool_impl_;
            pool_impl_ = nullptr;
        }

        if (weights_internal_ != nullptr) {
            delete weights_internal_;
            weights_internal_ = nullptr;
        }

        if (bias_internal_ != nullptr) {
            delete bias_internal_;
            bias_internal_ = nullptr;
        }

        release_buf();
    }

    virtual SaberStatus init(const std::vector<Tensor<X86> *> &inputs,
                             std::vector<Tensor<X86> *> &outputs,
                             ConvPoolingParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86> *> &inputs,
                               std::vector<Tensor<X86> *> &outputs,
                               ConvPoolingParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *> &inputs,
                                 std::vector<Tensor<X86> *> &outputs,
                                 ConvPoolingParam<X86> &param) override;

private:
    SaberStatus prepare_buf(Shape pool_shape, PoolingParam<X86> &pool_param, std::vector<float> scale);
    SaberStatus allocate_buf(Shape buf_shape, std::vector<float> scale);
    void release_buf();
    SaberStatus dispatch_conv(const std::vector<Tensor<X86> *> &inputs,
                            std::vector<Tensor<X86> *> &outputs,
                            ConvPoolingParam<X86> &param);
    SaberStatus dispatch_pool(std::vector<Tensor<X86> *> &inputs,
                            std::vector<Tensor<X86> *> &outputs,
                            PoolingParam<X86> &param);
    SaberStatus create_conv(const std::vector<Tensor<X86> *> &inputs,
                               std::vector<Tensor<X86> *> &outputs,
                               ConvPoolingParam<X86> &param,
                               Context<X86> &ctx);
    SaberStatus create_pool(std::vector<Tensor<X86> *> &inputs,
                               std::vector<Tensor<X86> *> &outputs,
                               PoolingParam<X86> &param,
                               Context<X86> &ctx);

    jit_avx512_core_u8s8s32x_conv_act_pool_kernel   *conv_kernel_;
    std::vector<Tensor<X86> *>                      buf_;
    Impl_pool_t                         *pool_impl_;
    Tensor<X86>                         *weights_internal_;
    Tensor<X86>                         *bias_internal_;
    Tensor<X86> _weights_scale;

    // quantization scale(s)
    std::vector<float> scale_;
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CONV_POOL_OPTIMIZED_H
