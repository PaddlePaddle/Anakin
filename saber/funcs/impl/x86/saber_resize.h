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

#ifndef ANAKIN_SABER_FUNCS_X86_SABER_RESIZE_H
#define ANAKIN_SABER_FUNCS_X86_SABER_RESIZE_H

#include "saber/funcs/impl/impl_resize.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberResize<X86, OpDtype>:
    public ImplBase<X86, OpDtype,ResizeParam<X86>> {

public:
    typedef Tensor<X86> DataTensor_in;
    typedef Tensor<X86> DataTensor_out;

    SaberResize() = default;
    ~SaberResize() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             ResizeParam<X86> &param,
                             Context<X86> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                               std::vector<DataTensor_out *>& outputs,
                               ResizeParam<X86> &param,
                               Context<X86> &ctx) {
        // do nothing
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in *>& inputs,
                                 std::vector<DataTensor_out *>& outputs,
                                 ResizeParam<X86> &param);


};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_X86_SABER_RESIZE_H
