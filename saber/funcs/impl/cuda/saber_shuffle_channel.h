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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SHUFFLE_CHANNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SHUFFLE_CHANNEL_H

#include "saber/funcs/impl/impl_shuffle_channel.h"
namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberShuffleChannel<NV, OpDtype> : \
    public ImplBase<NV, OpDtype, ShuffleChannelParam<NV>> {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberShuffleChannel() = default;
    ~SaberShuffleChannel() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             ShuffleChannelParam<NV>& param,
                             Context<NV> &ctx) {
        // get context
        this->_ctx = &ctx;
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               ShuffleChannelParam<NV>& param,
                               Context<NV>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 ShuffleChannelParam<NV>& param);

};
template class SaberShuffleChannel<NV, AK_FLOAT>;
template class SaberShuffleChannel<NV, AK_INT8>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SHUFFLE_CHANNEL_H
