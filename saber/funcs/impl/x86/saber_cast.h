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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CAST_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CAST_H

#include "anakin_config.h"
#include "saber/funcs/impl/impl_cast.h"
#include "saber/core/tensor.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberCast<X86, OpDtype> : \
    public ImplBase<
        X86,
        OpDtype,
        CastParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberCast() = default;
    ~SaberCast() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                      std::vector<Tensor<X86>*>& outputs,
                      CastParam<X86> &param, Context<X86> &ctx){
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                        std::vector<Tensor<X86>*>& outputs,
                        CastParam<X86> &param, Context<X86> &ctx){

        _inDtype = param.in_type;
        _outDtype = param.out_type;
        if(_inDtype != 1 && _inDtype !=5){// AK_FLOAT AK_INT32
            LOG(FATAL) << "Cast not impl other type: " << _inDtype;
        }
        if(_outDtype != 1 && _outDtype !=5){
            LOG(FATAL) << "Cast not impl other type: " << _outDtype;
        }
        CHECK_EQ(_inDtype, inputs[0]->get_dtype()) << "inputs data type should be same with param.in_type";
        CHECK_EQ(_outDtype, outputs[0]->get_dtype()) << "outputs data type should be same with param.out_type";
        
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                          std::vector<Tensor<X86>*>& outputs,
                          CastParam<X86> &param)override;

private:
    int _inDtype;
    int _outDtype;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_Cast_H
