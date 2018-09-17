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

#ifndef ANAKIN_SABER_FUNCS_X86_SABER_SLICE_H
#define ANAKIN_SABER_FUNCS_X86_SABER_SLICE_H

#include "saber/funcs/impl/impl_slice.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSlice<X86, OpDtype>:
    public ImplBase<X86, OpDtype, SliceParam<X86>> {

public:

    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberSlice() = default;
    ~SaberSlice() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             SliceParam<X86> &param,
                             Context<X86> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

   virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               SliceParam<X86> &param,
                               Context<X86> &ctx) {

        _slice_num = inputs[0]->count_valid(0, param.axis);
        _slice_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 SliceParam<X86> &param);

private:
    int _slice_num;
    int _slice_size;

};
template class SaberSlice<X86, AK_FLOAT>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_X86_SABER_SLICE_H
