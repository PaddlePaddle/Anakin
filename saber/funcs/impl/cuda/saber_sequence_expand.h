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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_EXPAND_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_EXPAND_H

#include "saber/funcs/impl/impl_sequence_expand.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberSequenceExpand<NV, OpDtype> : \
    public ImplBase <
        NV, OpDtype, SequenceExpandParam<NV> > {
public:
    typedef Tensor<NV> OpTensor;
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberSequenceExpand()
    {}

    ~SaberSequenceExpand() {}

    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             SequenceExpandParam<NV>& param, Context<NV>& ctx) {
        this->_ctx = &ctx;
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               SequenceExpandParam<NV>& param, Context<NV>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 SequenceExpandParam<NV>& param);
private:
    Tensor<NV> _seq_id_map;

};

//template class SaberSequenceExpand<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SEQUENCE_EXPAND_H