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
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
#include "cuda_utils.h"
namespace anakin {

namespace saber {

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class SaberLstm<NV, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>: \
    public ImplBase <
    Tensor<NV, inDtype, LayOutType_in>, \
    Tensor<NV, outDtype, LayOutType_out>, \
    Tensor<NV, OpDtype, LayOutType_op>, \
    LstmParam<Tensor<NV, OpDtype, LayOutType_op> >> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberLstm() {}
    ~SaberLstm() {

    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs, \
                             std::vector<DataTensor_out*>& outputs, \
                             LstmParam <OpTensor>& param, Context<NV>& ctx) {

        this->_ctx = &ctx;
        if(param.with_peephole){
            _hidden_size=param.bias()->valid_size()/7;
        }else{
            _hidden_size=param.bias()->valid_size()/4;
        }
        _word_size=(param.weight()->valid_size()-_hidden_size*_hidden_size*4)/_hidden_size/4;

        _seq_util = SeqSortedseqTranseUtil(param.is_reverse);
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs, \
                               std::vector<DataTensor_out*>& outputs, \
                               LstmParam<OpTensor>& param, Context<NV>& ctx) {
        if (!(&ctx == this->_ctx)) {
            this->_ctx = &ctx;
        }
        int batch_size = inputs[0]->get_seq_offset().size() - 1;
        if (batch_size > 0) {
            int sequence = inputs[0]->num();
            _gemm_wx = saber_find_fast_sass_gemm(false, false, sequence, 4 * _hidden_size, _word_size);
            _gemm_wh = saber_find_fast_sass_gemm(false, false, batch_size, 4 * _hidden_size, _hidden_size);
        }
        return SaberSuccess;
    }


    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 LstmParam <OpTensor>& param);

private:
    int _word_size;
    int _hidden_size;

    OpTensor _init_hidden;

    OpTensor _temp_wx;
    OpTensor _temp_wh;
    OpTensor _temp_cell;

    OpTensor _temp_x;
    OpTensor _temp_out;
    OpTensor _temp_h_init;


    OpTensor _temp_map_dev;
    OpTensor _temp_zero;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wx;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wh;

    SeqSortedseqTranseUtil _seq_util;

    SaberStatus
    dispatch_batch(
            const std::vector < DataTensor_in* >& inputs,
            std::vector < DataTensor_out* >& outputs,
            LstmParam < OpTensor >& param);

    SaberStatus
    dispatch_once(
            const std::vector < DataTensor_in* >& inputs,
            std::vector < DataTensor_out* >& outputs,
            LstmParam < OpTensor >& param);

};



} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GRU_H
