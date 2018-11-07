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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_UPADDING_PADDING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_UPADDING_PADDING_H

#include "saber/funcs/impl/impl_conv_unpadding_padding.h"
#include "saber/funcs/saber_util.h"
namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberConvUnpaddingPadding<NV, OpDtype> : \
    public ImplBase <
    NV, OpDtype,
    ConvUnpaddingPaddingParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberConvUnpaddingPadding()
    {}

    ~SaberConvUnpaddingPadding() {

    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             ConvUnpaddingPaddingParam<NV>& param,
                             Context<NV>& ctx) {
        this->_ctx = &ctx;
        _width_offset_tensor.set_dtype(AK_INT32);
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               ConvUnpaddingPaddingParam<NV>& param,
                               Context<NV>& ctx) {
        this->_ctx = &ctx;
//        std::vector<int> width_vector = inputs[0]->get_seq_offset()[0];
//        utils::try_expand_tensor(_width_offset_tensor, width_vector.size());
//        CUDA_CHECK(cudaMemcpyAsync(_width_offset_tensor.mutable_data(), width_vector.data(),
//                                   sizeof(int)*width_vector.size(), cudaMemcpyHostToDevice, this->_ctx->get_compute_stream()));
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 ConvUnpaddingPaddingParam<NV>& param);

private:
    Tensor<NV> _width_offset_tensor;

};


}

}

#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
