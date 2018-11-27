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


#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_POOLING_H

#include "saber/funcs/impl/impl_pooling.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_pool_kernel_f32.h"
#include "saber/funcs/impl/x86/kernel/jit_generator.h"

namespace anakin{
namespace saber {

using namespace jit;

template <DataType OpDtype>
class SaberPooling<X86, OpDtype> : public ImplBase<
        X86,
        OpDtype,
        PoolingParam<X86>>
{
public:
    typedef Tensor<X86> DataTensor_in;
    typedef Tensor<X86> DataTensor_out;

    SaberPooling()
            : _kernel(nullptr) {}

    ~SaberPooling() {
        if (_kernel != nullptr) {
            delete _kernel;
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             PoolingParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               PoolingParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 PoolingParam<X86> &param) override;

    virtual SaberStatus init_conf(jit_pool_conf_t &jpp,
                                  const std::vector<DataTensor_in*>& inputs,
                                  std::vector<DataTensor_out*>& outputs,
                                  PoolingParam<X86>& param);
private:
    jit_uni_pool_kernel_f32<avx512_common>* _kernel;
};


}
}

#endif
