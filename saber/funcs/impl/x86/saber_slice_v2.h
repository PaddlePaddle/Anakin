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

#ifndef ANAKIN_SABER_FUNCS_X86_SABER_SLICE_V2_H
#define ANAKIN_SABER_FUNCS_X86_SABER_SLICE_V2_H

#include "saber/funcs/impl/impl_slice_v2.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSliceV2<X86, OpDtype>:
    public ImplBase<X86, OpDtype, SliceV2Param<X86>> {

public:

    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberSliceV2() = default;
    ~SaberSliceV2() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             SliceV2Param<X86> &param,
                             Context<X86> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

   virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               SliceV2Param<X86> &param,
                               Context<X86> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 SliceV2Param<X86> &param);

private:
    std::vector<int> _starts;
    std::vector<int> _ends;
    std::vector<int> _axes;

};
template class SaberSliceV2<X86, AK_FLOAT>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_X86_SABER_SLICE_V2_H
