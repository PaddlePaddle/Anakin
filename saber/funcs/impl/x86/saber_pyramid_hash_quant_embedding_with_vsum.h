/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PYRAMID_HASH_QUANT_EMBEDDING_WITH_VSUM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PYRAMID_HASH_QUANT_EMBEDDING_WITH_VSUM_H

#include "saber/funcs/impl/impl_pyramid_hash_quant_embedding_with_vsum.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberPyramidHashQuantEmbeddingWithVsum<X86, OpDtype> :
    public ImplBase<
        X86, OpDtype,
        PyramidHashQuantEmbeddingParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberPyramidHashQuantEmbeddingWithVsum() {}

    ~SaberPyramidHashQuantEmbeddingWithVsum() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             PyramidHashQuantEmbeddingParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               PyramidHashQuantEmbeddingParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 PyramidHashQuantEmbeddingParam<X86> &param) override;
    virtual SaberStatus hash_embedding_forward(const OpDataType* buffer,
                           int len,
                           const OpDataType* quant_dict,
                           const unsigned char* weights,
                           OpDataType* out);

private:
    int _space_size;
    int _emb_size;
    int _pyramid_layer;
    int _rand_len;
    int _white_filter_size;
    int _black_filter_size;
    float _dropout_percent;
    int _quant_bit;
    int _dict_size;
};

}
}
#endif
