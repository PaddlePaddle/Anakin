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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_CONCAT_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_CONCAT_H

#include "saber/funcs/impl/impl_concat.h"
#include "saber/core/tensor.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberConcat<ARM, OpDtype> : \
    public ImplBase<
        ARM, OpDtype,
        ConcatParam<ARM> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberConcat() = default;
    ~SaberConcat() {}

    virtual SaberStatus init(const std::vector<Tensor<ARM> *>& inputs,
                      std::vector<Tensor<ARM> *>& outputs,
                      ConcatParam<ARM> &param, Context<ARM> &ctx){
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConcatParam<ARM> &param, Context<ARM> &ctx){

        _num_concats = inputs[0]->count_valid(0, param.axis);
        _concat_input_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<ARM> *>& inputs,
                          std::vector<Tensor<ARM> *>& outputs,
                          ConcatParam<ARM> &param);

private:
    int _num_concats;
    int _concat_input_size;
};

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_CONCAT_H
