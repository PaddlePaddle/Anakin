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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_REDUCE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_REDUCE_H

#include "saber/funcs/impl/impl_reduce.h"
#include <functional>
#include <map>

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberReduce<NV, OpDtype> :
        public ImplBase<
                NV, OpDtype,
                ReduceParam<NV> > {
public:
    typedef ImplBase<NV, OpDtype, ReduceParam<NV> > Impl_t;
    SaberReduce() = default;
    ~SaberReduce() {
        delete _impl;
    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
            std::vector<Tensor<NV> *>& outputs,
            ReduceParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
            std::vector<Tensor<NV> *>& outputs,
            ReduceParam<NV>& param, Context<NV> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
            std::vector<Tensor<NV>*>& outputs,
            ReduceParam<NV>& param);

private:
    Buffer<NV> _rdim_b;
    Buffer<NV> _ndim_b;
    Buffer<NV> _i_stride_b;
    Buffer<NV> _o_stride_b;
    Impl_t* _impl{nullptr};
    typedef void reduce_kernel(
            const float*, float*, const int*, const int*,
            const int*, const int*, int);
    std::map<ReduceType,
        std::vector<std::vector<reduce_kernel*>>> _kernel_direct_map;
    bool _template_reduction{false};
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_REDUCE_H
