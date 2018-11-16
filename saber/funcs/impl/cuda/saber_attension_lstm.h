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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ATTENSION_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ATTENSION_LSTM_H
#include "saber/funcs/impl/impl_attension_lstm.h"
namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberAttensionLstm<NV, OpDtype>: public ImplBase <
        NV, OpDtype, AttensionLstmParam<NV >> {

public:
    typedef Tensor<NV> OpTensor;
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberAttensionLstm() {}
    ~SaberAttensionLstm() {
        for (int i = 0; i < _attn_outs.size(); i++) {
            delete _attn_outs[i];
            _attn_outs[i] = nullptr;
        }
    }

    virtual SaberStatus init(const std::vector<OpTensor*>& inputs, \
                             std::vector<OpTensor*>& outputs, \
                             AttensionLstmParam <NV>& attension_lstm_param, Context<NV>& ctx);

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs, \
                               std::vector<OpTensor*>& outputs, \
                               AttensionLstmParam<NV>& attension_lstm_param, Context<NV>& ctx) ;


    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 AttensionLstmParam <NV>& param);

private:
    Tensor<NV> _dev_offset;
    Tensor<NV> _dev_seq_id_map;
    std::vector<OpTensor*> _attn_outs;
    OpTensor _first_fc_out_0;
    OpTensor _first_fc_out_1;
    OpTensor _softmax_out;
    OpTensor _pool_out;
    int _word_size;
    int _hidden_size;
    OpTensor _hidden_out;
    OpTensor _cell_out;
    OpTensor _lstm_out;
    int _max_seq_len;
    cublasHandle_t _handle;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wx;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wh;

    typedef std::function<OpDataType(OpDataType)> ActFunction;

};



} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ATTENSION_LSTM_H