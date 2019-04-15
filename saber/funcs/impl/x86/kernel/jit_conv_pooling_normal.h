/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_CONV_POOLING_NORMAL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_CONV_POOLING_NORMAL_H

#include "anakin_config.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_macro.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class JitConvPoolingNormal : public ImplBase<
        X86, OpDtype, ConvPoolingParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    typedef ImplBase<X86, OpDtype, ConvParam<X86> > Impl_conv_t;
    typedef ImplBase<X86, OpDtype, PoolingParam<X86> > Impl_pool_t;

    JitConvPoolingNormal()
        : conv_impl_(nullptr)
        , pool_impl_(nullptr){
    }

    ~JitConvPoolingNormal() {
        if (conv_impl_ != nullptr) {
            delete conv_impl_;
            conv_impl_ = nullptr;
        }
        if (pool_impl_ != nullptr) {
            delete pool_impl_;
            pool_impl_ = nullptr;
        }

        release_buf();
    }

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             ConvPoolingParam<X86>& param, Context<X86> &ctx);

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               ConvPoolingParam<X86>& param, Context<X86>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 ConvPoolingParam<X86>& param);

private:
    SaberStatus prepare_buf(Shape pool_shape, PoolingParam<X86> pool_param, std::vector<float> scale);
    SaberStatus allocate_buf(Shape buf_shape, std::vector<float> scale);
    void release_buf();

    Impl_conv_t* conv_impl_;
    Impl_pool_t* pool_impl_;

    std::vector<Tensor<X86> *> buf_;
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_JIT_CONV_POOLING_NORMAL_H
