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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_ATTENSION_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_ATTENSION_LSTM_H


#include "saber/funcs/impl/impl_attension_lstm.h"
#include "saber/funcs/impl/x86/x86_utils.h"


namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberAttensionLstm<X86, OpDtype>: public ImplBase <
        X86, OpDtype, AttensionLstmParam<X86 >> {
public:
    typedef Tensor<X86> OpTensor;
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberAttensionLstm(): _hidden_size(0) {};

    ~SaberAttensionLstm() {};

    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             AttensionLstmParam<X86>& param,
                             Context<X86>& ctx) ;

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               AttensionLstmParam<X86>& param,
                               Context<X86>& ctx) {
        return SaberSuccess;
    };

    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 AttensionLstmParam<X86>& param) ;



private:

    int _word_size;
    int _hidden_size;

    const OpDataType*  _weights_i2h;
    const OpDataType*  _weights_h2h;
    const OpDataType*  _weights_bias;

    std::vector<OpTensor*> _attn_fc_weights;
    std::vector<OpTensor*> _attn_fc_bias;
    std::vector<int> _attn_fc_size;

    std::vector<OpTensor*> _attn_outs;
    OpTensor _first_fc_out_0;
    OpTensor _first_fc_out_1;
    OpTensor _softmax_out;
    OpTensor _pool_out;
    OpTensor _cell_out;
    OpTensor _hidden_out;
    OpTensor _lstm_out;
    int _max_seq_len;

    SaberStatus cpu_dispatch(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             AttensionLstmParam<X86>& param);
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_ATTENSION_LSTM_H
