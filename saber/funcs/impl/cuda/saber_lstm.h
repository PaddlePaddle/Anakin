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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LSTM_H
#include "saber/funcs/impl/impl_lstm.h"
#include "sass_funcs.h"
#include "cuda_utils.h"
namespace anakin {

namespace saber {

static int round_up(int k, int c) {
    return ((k + c - 1) / c) * c;
}

template <DataType OpDtype>
class SaberLstm<NV, OpDtype>: public ImplBase <
        NV, OpDtype,LstmParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberLstm() {}
    ~SaberLstm() {

    }

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs, \
                             std::vector<Tensor<NV>*>& outputs, \
                             LstmParam <NV>& param, Context<NV>& ctx) {

        this->_ctx = &ctx;
        if(param.with_peephole){
            _hidden_size=param.bias()->valid_size()/7;
        }else{
            _hidden_size=param.bias()->valid_size()/4;
        }
        _word_size=(param.weight()->valid_size()-_hidden_size*_hidden_size*4)/_hidden_size/4;
        //TODO:add round_up to saber_util
        _aligned_hidden_size=round_up(_hidden_size,32);


        _seq_util = SeqSortedseqTranseUtil(param.is_reverse);
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs, \
                               std::vector<Tensor<NV>*>& outputs, \
                               LstmParam < NV >& param, Context<NV>& ctx) {
        if (!(&ctx == this->_ctx)) {
            this->_ctx = &ctx;
        }

        std::vector<std::vector<int>> lod=inputs[0]->get_seq_offset();
        std::vector<int> offset=lod[lod.size()-1];
        int batch_size = offset.size() - 1;
        CHECK_GE(batch_size,1)<<"batchsize must >= 1";

        int sequence = inputs[0]->num();
        _gemm_wx = saber_find_fast_sass_gemm(false, false, sequence, 4 * _hidden_size,_word_size);
        _gemm_wh = saber_find_fast_sass_gemm(false, false, batch_size, 4 * _hidden_size, _hidden_size);
        return SaberSuccess;
    }


    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 LstmParam <NV>& param);

private:
    int _word_size;
    int _hidden_size;
    int _aligned_hidden_size;

    Tensor<NV> _init_hidden;

    Tensor<NV> _temp_wx;
    Tensor<NV> _temp_wh;
    Tensor<NV> _temp_cell;

    Tensor<NV> _temp_x;
    Tensor<NV> _temp_out;
    Tensor<NV> _temp_h_init;


    Tensor<NV> _temp_map_dev;
    Tensor<NV> _temp_zero;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wx;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wh;

    SeqSortedseqTranseUtil _seq_util;

    SaberStatus
    dispatch_batch(
            const std::vector < Tensor<NV>* >& inputs,
            std::vector < Tensor<NV>* >& outputs,
            LstmParam < NV >& param);

    SaberStatus
    dispatch_once(
            const std::vector < Tensor<NV>* >& inputs,
            std::vector < Tensor<NV>* >& outputs,
            LstmParam < NV >& param);

};



} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GRU_H
