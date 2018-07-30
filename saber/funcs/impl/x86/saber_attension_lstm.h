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

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class SaberAttensionLstm<X86, OpDtype, inDtype, outDtype,
          LayOutType_op, LayOutType_in, LayOutType_out>: public ImplBase <
          Tensor<X86, inDtype, LayOutType_in>,
          Tensor<X86, outDtype, LayOutType_out>,
          Tensor<X86, OpDtype, LayOutType_op>,
          AttensionLstmParam<Tensor<X86, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    SaberAttensionLstm():_hidden_size(0){};

    ~SaberAttensionLstm() {};

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             AttensionLstmParam<OpTensor>& param,
                             Context<X86>& ctx) {
        utils::AlignedUtils aligned_tool;
        auto lstm_param = param.lstm_param;
        auto attn_param = param.attension_param;
        
        _hidden_size = lstm_param.bias()->valid_size() / 4;
        _word_size = lstm_param.weight()->valid_size() / (4 * _hidden_size)  -  _hidden_size;

        int weights_i2h_size = 4 * _hidden_size * _word_size;
        _weights_i2h = lstm_param.weight()->data();
        _weights_h2h = lstm_param.weight()->data() + weights_i2h_size;
        _weights_bias = lstm_param.bias()->data();

        int input_dim = inputs[0]->valid_size() / inputs[0]->num();
        int fc_num = attn_param.fc_vec.size();
        _attn_fc_weights.resize(fc_num);
        _attn_fc_bias.resize(fc_num);
        _attn_fc_size.resize(fc_num);
        _attn_outs.resize(fc_num);
        for (int i = 0; i < fc_num; i++) {
            int N = attn_param.fc_vec[i].num_output;
            auto fc_param = (attn_param.fc_vec[i]);
            _attn_fc_weights[i] = fc_param.weights;
            _attn_fc_bias[i] = fc_param.bias;
            _attn_fc_size[i] = N;
            Shape shape = {1, 1, 1, 1};
            _attn_outs[i] = new DataTensor_out(shape);
        }

        return SaberSuccess;
    };

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               AttensionLstmParam<OpTensor>& param,
                               Context<X86>& ctx) {return SaberSuccess;};

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 AttensionLstmParam<OpTensor>& param) ;



private:

    int _word_size;
    int _hidden_size;

    DataType_op*  _weights_i2h;
    DataType_op*  _weights_h2h;
    DataType_op*  _weights_bias;

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

    SaberStatus cpu_dispatch(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             AttensionLstmParam<OpTensor>& param);
};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_ATTENSION_LSTM_H
