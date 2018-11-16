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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_EXPAND_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_EXPAND_H

#include "saber/funcs/impl/impl_expand.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberExpand<NV, OpDtype>: public ImplBase<NV, OpDtype, ExpandParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberExpand() {}
    ~SaberExpand() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             ExpandParam<NV> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               ExpandParam<NV> &param,
                               Context<NV> &ctx) {
        int dims = param.expand_times.size();
        Shape shape = std::vector<int>({dims, 1, 1, 1});
        _in_shape.re_alloc(shape, AK_INT32);
        _expand_times.re_alloc(shape, AK_INT32);
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 ExpandParam<NV> &param);

private:
    Tensor<NV> _in_shape;
    Tensor<NV> _expand_times;
};

template class SaberExpand<NV, AK_FLOAT>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_EXPAND_H
