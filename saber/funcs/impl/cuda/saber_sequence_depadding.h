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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_DEPADDING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_DEPADDING_H

#include "saber/funcs/impl/impl_sequence_depadding.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberSequenceDePadding<NV, OpDtype> :
    public ImplBase<
        NV, OpDtype,
        SequenceDePaddingParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    SaberSequenceDePadding() = default;
    ~SaberSequenceDePadding() = default;

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            SequenceDePaddingParam<NV>& param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            SequenceDePaddingParam<NV>& param, Context<NV> &ctx);
    
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                          std::vector<Tensor<NV>*>& outputs,
                          SequenceDePaddingParam<NV>& param);
private:
    Tensor<NV> _seq_id_map;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_DEPADDING_H
