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
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
namespace anakin {

namespace saber {

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class SaberAttensionLstm<NV, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>: \
    public ImplBase <
    Tensor<NV, inDtype, LayOutType_in>, \
    Tensor<NV, outDtype, LayOutType_out>, \
    Tensor<NV, OpDtype, LayOutType_op>, \
    AttensionLstmParam<Tensor<NV, OpDtype, LayOutType_op> >> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberAttensionLstm() {}
    ~SaberAttensionLstm() {
        for (int i = 0; i < _attn_outs.size(); i++) {
            delete _attn_outs[i];
            _attn_outs[i] = nullptr;
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs, \
                             std::vector<DataTensor_out*>& outputs, \
                             AttensionLstmParam <OpTensor>& attension_lstm_param, Context<NV>& ctx) {

        this->_ctx = &ctx;
        auto cuda_stream = ctx.get_compute_stream();

        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, cuda_stream));

        auto lstm_param = attension_lstm_param.lstm_param;
        _hidden_size = lstm_param.bias()->valid_size() / 4 / lstm_param.num_layers;
        int weights_h2h_size = _hidden_size * _hidden_size * 4 * (2 * lstm_param.num_layers - 1);
        int weights_i2h_size = lstm_param.weight()->valid_size() - weights_h2h_size;
        _word_size = weights_i2h_size / (4 * _hidden_size);
        auto fc_vec = attension_lstm_param.attension_param.fc_vec;
        _attn_outs.resize(fc_vec.size());
        _max_seq_len = 100;
        for (int i = 0; i < fc_vec.size(); i++) {
           Shape shape = {inputs[0]->num(), fc_vec[i].num_output, 1, 1};
           _attn_outs[i] = new DataTensor_out(shape);
        }

        return create(inputs, outputs, attension_lstm_param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs, \
                               std::vector<DataTensor_out*>& outputs, \
                               AttensionLstmParam<OpTensor>& attension_lstm_param, Context<NV>& ctx) {

        int batch_size = inputs[0]->get_seq_offset().size() - 1;
        int sequence = inputs[0]->num();

        _gemm_wx = saber_find_fast_sass_gemm(false, false, 
               sequence, 4 * _hidden_size, _word_size);
        _gemm_wh = saber_find_fast_sass_gemm(false, false, batch_size, 4 * _hidden_size, _hidden_size);

        return SaberSuccess;
    }


    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 AttensionLstmParam <OpTensor>& param);

private:
    Tensor<NV, AK_INT32, LayOutType_in> _dev_offset;
    Tensor<NV, AK_INT32, LayOutType_in> _dev_seq_id_map;
    std::vector<DataTensor_out*> _attn_outs;
    DataTensor_out _first_fc_out_0;
    DataTensor_out _first_fc_out_1;
    DataTensor_out _softmax_out;
    DataTensor_out _pool_out;
    int _word_size;
    int _hidden_size;
    DataTensor_out _hidden_out;
    DataTensor_out _cell_out;
    DataTensor_out _lstm_out;
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
