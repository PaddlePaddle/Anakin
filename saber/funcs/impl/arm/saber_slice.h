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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SLICE_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SLICE_H

#include "saber/funcs/impl/impl_slice.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSlice<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        SliceParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberSlice()
    {}

    ~SaberSlice() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            SliceParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            SliceParam<ARM>& param, Context<ARM> &ctx) {
        _slice_num = inputs[0]->count_valid(0, param.axis);
        _slice_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          SliceParam<ARM>& param);
private:
    int _slice_num = 4;
    int _slice_size = 0;
    std::vector<int> _slice_points;

    Tensor<ARM> _tmp_in;
    Tensor<ARM> _tmp_out;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Slice_H
