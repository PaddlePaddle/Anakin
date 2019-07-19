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

#ifndef ANAKIN_SABER_FUNCS_CUDA_SABER_SLICE_V2_H
#define ANAKIN_SABER_FUNCS_CUDA_SABER_SLICE_V2_H

#include "saber/funcs/impl/impl_slice_v2.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSliceV2<NV, OpDtype>:
    public ImplBase<NV, OpDtype, SliceV2Param<NV>> {

public:

    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberSliceV2() = default;
    ~SaberSliceV2() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             SliceV2Param<NV> &param,
                             Context<NV> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

   virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               SliceV2Param<NV> &param,
                               Context<NV> &ctx);


    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 SliceV2Param<NV> &param);

private:
    Tensor<NV> _starts_d;
    Tensor<NV> _in_stride_d;
    Tensor<NV> _out_shape_d;
    Tensor<NV> _axes_d;

};
template class SaberSliceV2<NV, AK_FLOAT>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_CUDA_SABER_SLICE_H
