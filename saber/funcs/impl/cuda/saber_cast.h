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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CAST_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CAST_H

#include "saber/funcs/impl/impl_cast.h"

namespace anakin{

namespace saber{

template < DataType OpDtype>
class SaberCast<NV, OpDtype> : \
    public ImplBase<
       NV, OpDtype,
        CastParam<NV> > 
{
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberCast()
    {}

    ~SaberCast() {

    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            CastParam<NV>& param, Context<NV>& ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            CastParam<NV>& param, Context<NV> &ctx) {
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
    
    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                          std::vector<Tensor<NV> *>& outputs,
                          CastParam<NV>& param);

    private:
    int _inDtype;
    int _outDtype;
};

//template class SaberCast<NV, AK_FLOAT, AK_FLOAT, AK_INT32, NCHW, NCHW, NCHW>;
//template class SaberCast<NV, AK_FLOAT, AK_INT32, AK_FLOAT, NCHW, NCHW, NCHW>;

}

}

#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
