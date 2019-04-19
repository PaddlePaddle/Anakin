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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_CAST_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_CAST_H

#include "saber/funcs/impl/impl_cast.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberCast<ARM, OpDtype> : \
    public ImplBase<
        ARM,
        OpDtype,
        CastParam<ARM > >
{
public:
    typedef typename DataTrait<ARM, OpDtype>::Dtype OpDataType;

    SaberCast()
    {}

    ~SaberCast() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            CastParam<ARM>& param, Context<ARM>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            CastParam<ARM>& param, Context<ARM> &ctx) {
        bool in_f = (param.in_type == AK_FLOAT) || (param.in_type == AK_INT32);
        bool out_f = (param.in_type == AK_FLOAT) || (param.in_type == AK_INT32);
        bool eq_f = param.in_type == param.out_type;
        if (!(in_f && out_f || eq_f)){
            LOG(ERROR) << "only support in_type, out_type is AK_FLOAT or AK_INT32 or in_type == out_type";
        }
        CHECK_EQ(inputs[0]->get_dtype(), param.in_type) << "input data type should be same with param.in_type";
        CHECK_EQ(outputs[0]->get_dtype(), param.out_type) << "output data type should be same with param.out_type";
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          CastParam<ARM>& param);
private:
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_Cast_H
