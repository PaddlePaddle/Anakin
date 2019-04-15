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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PRODUCT_QUANT_EMBEDDING_WITH_VSUM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PRODUCT_QUANT_EMBEDDING_WITH_VSUM_H

#include "saber/funcs/impl/impl_product_quant_embedding_with_vsum.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberProductQuantEmbeddingWithVsum<X86, OpDtype> :
    public ImplBase<
        X86, OpDtype,
        ProductQuantEmbeddingWithVsumParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberProductQuantEmbeddingWithVsum() {}

    ~SaberProductQuantEmbeddingWithVsum() {
        delete [] _buf;
    }

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             ProductQuantEmbeddingWithVsumParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               ProductQuantEmbeddingWithVsumParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 ProductQuantEmbeddingWithVsumParam<X86> &param) override;

private:
    int _voc_size;
    int _emb_size;
    int _max_seq_len;
    int _unigram_num[3];
    int  _bigram_num[3];
    int _collocation_num[3];
    int _chnl_num[3];
    int _word_len[3];
    int _word_num[3];
    int _dict_size[3];
    int _word_offset[9];
    int _real_offset[9];
    const unsigned char* _weights[3];
    const float* _quant_dict[3];
    
    unsigned int* _buf;
};

}
}
#endif
