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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GRU_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GRU_H
#include "saber/funcs/impl/impl_gru.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
#include "cuda_utils.h"
namespace anakin {

namespace saber {



template <DataType OpDtype>
class SaberGru<NV, OpDtype>: public ImplBase <
        NV, OpDtype,GruParam<NV> > {

public:

    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    typedef Tensor<NV> OpTensor;
    SaberGru() {}
    ~SaberGru() {

    }

    virtual SaberStatus init(const std::vector<OpTensor*>& inputs, \
                             std::vector<OpTensor*>& outputs, \
                             GruParam <NV>& param, Context<NV>& ctx) {

        this->_ctx = &ctx;

        CHECK(param.init_hidden() == nullptr)<< "only support init_hidden == null now";
        CHECK_EQ(param.formula ,GRU_ORIGIN)<<"only support gru_origin now";

        _hidden_size = param.bias()->valid_size() / 3;

        int weights_h2h_size = _hidden_size * _hidden_size * 3;
        int weights_i2h_size = param.weight()->valid_size() - weights_h2h_size;

        _word_size = weights_i2h_size / _hidden_size / 3;

        _seq_util = SeqSortedseqTranseUtil(param.is_reverse);

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs, \
                               std::vector<OpTensor*>& outputs, \
                               GruParam<NV>& param, Context<NV>& ctx) {

        if (!(&ctx == this->_ctx)) {
            this->_ctx = &ctx;
        }

        int batch_size = inputs[0]->get_seq_offset().size() - 1;
        int sequence = inputs[0]->num();
        _gemm_wx = saber_find_fast_sass_gemm(false, false, sequence, 3 * _hidden_size,
                                             _word_size);
        _gemm_wh_2 = saber_find_fast_sass_gemm(false, false, batch_size, 2 * _hidden_size, _hidden_size);

        _gemm_wh_o = saber_find_fast_sass_gemm(false, false, batch_size, 1 * _hidden_size, _hidden_size);
        return SaberSuccess;
    }


    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 GruParam <NV>& param);

private:

    /**
     * for hw2seq
     */
    OpTensor _temp_tensor_in;
    OpTensor _temp_tensor_out;
    OpTensor _temp_wx;
    OpTensor _temp_wh;
    OpTensor _temp_whr;

    OpTensor _temp_zero;

    OpTensor _temp_vector_offset;
    OpTensor _temp_map_host;
    OpTensor _temp_map_dev;

    SeqSortedseqTranseUtil _seq_util;
    int _word_size;
    int _hidden_size;


    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wx;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wh_2;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wh_o;


};



} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GRU_H