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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H


#include "saber/funcs/impl/impl_lstm.h"
#include "saber/funcs/impl/x86/x86_utils.h"


#ifdef __AVX512F__
#include "saber_avx512_activation.h"
#define SABER_X86_TYPE __m512
#elif __AVX2__
#include "saber_avx2_activation.h"
#define SABER_X86_TYPE __m256
#else
#include "saber_normal_activation.h"
#define SABER_X86_TYPE float
#endif

namespace anakin {
namespace saber {

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class SaberLstm<X86, OpDtype, inDtype, outDtype,
          LayOutType_op, LayOutType_in, LayOutType_out>: public ImplBase <
          Tensor<X86, inDtype, LayOutType_in>,
          Tensor<X86, outDtype, LayOutType_out>,
          Tensor<X86, OpDtype, LayOutType_op>,
          LstmParam<Tensor<X86, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    SaberLstm():_hidden_size(0){};

    ~SaberLstm() {};

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             LstmParam<OpTensor>& param,
                             Context<X86>& ctx) {
        if(param.with_peephole){
            _hidden_size=param.bias()->valid_size()/7;
        }else{
            _hidden_size=param.bias()->valid_size()/4;
        }
        _word_size=(param.weight()->valid_size()-_hidden_size*_hidden_size*4)/_hidden_size/4;

        LOG(INFO)<<"wordsize = "<<_word_size;
        int weights_i2h_size=4*_hidden_size*_word_size;
        int weights_h2h_size=4*_hidden_size*_hidden_size;
        int weights_bias_size=4*_hidden_size;
        int weights_peephole_size=3*_hidden_size;

        int aligned_byte= sizeof(SABER_X86_TYPE);
        int c_size=aligned_byte/sizeof(DataType_op);

        _aligned_word_size=utils::round_up(_word_size,c_size);
        _aligned_hidden_size=utils::round_up(_hidden_size,c_size);


        Shape aligned_weights_i2h_shape(1,_word_size,4,_aligned_hidden_size);
        Shape aligned_weights_h2h_shape(1,_aligned_hidden_size,4,_aligned_hidden_size);
        Shape aligned_weights_bias_shape(1,1,4,_aligned_hidden_size);
        _aligned_weights_i2h.try_expand_size(aligned_weights_i2h_shape);
        _aligned_weights_h2h.try_expand_size(aligned_weights_h2h_shape);
        _aligned_weights_bias.try_expand_size(aligned_weights_bias_shape);

        utils::AlignedUtils aligned_tool;
        aligned_tool.aligned_last_dim(param.weight()->data(),_aligned_weights_i2h.mutable_data(),
                                      weights_i2h_size,_hidden_size,_aligned_hidden_size);

        aligned_tool.aligned_last_dim(param.weight()->data() + weights_i2h_size,_aligned_weights_h2h.mutable_data(),
                                      weights_h2h_size,_hidden_size,_aligned_hidden_size);

        aligned_tool.aligned_last_dim(param.bias()->data(),_aligned_weights_bias.mutable_data(),
                                      weights_bias_size,_hidden_size,_aligned_hidden_size);
//FIXME:init weights tensor
        if(param.with_peephole){
            Shape aligned_weights_peephole_shape(1,1,3,_aligned_hidden_size);
            _aligned_weights_peephole.try_expand_size(aligned_weights_peephole_shape);
            aligned_tool.aligned_last_dim(param.bias()->data()+weights_bias_size,_aligned_weights_peephole.mutable_data(),
                                          weights_peephole_size,_hidden_size,_aligned_hidden_size);
        }

        return SaberSuccess;
    };

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               LstmParam<OpTensor>& param,
                               Context<X86>& ctx) {return SaberSuccess;};

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 LstmParam<OpTensor>& param) ;



private:

    int _word_size;
    int _hidden_size;
    int _aligned_word_size;
    int _aligned_hidden_size;


    OpTensor _weights_i2h;
    OpTensor _weights_h2h;
    OpTensor _weights_bias;
    OpTensor _weights_peephole;
    OpTensor _init_hidden;

    OpTensor _aligned_weights_i2h;
    OpTensor _aligned_weights_h2h;
    OpTensor _aligned_weights_bias;
    OpTensor _aligned_weights_peephole;

    OpTensor _aligned_init_hidden;

    OpTensor _temp_wx;
    OpTensor _temp_wh;
    OpTensor _temp_cell;

    OpTensor _temp_x;
    OpTensor _temp_out;
    OpTensor _temp_h_init;

    template <typename BIT>
    SaberStatus avx_dispatch_without_peephole(const std::vector<DataTensor_in*>& inputs,
                                           std::vector<DataTensor_out*>& outputs,
                                           LstmParam<OpTensor>& param);

    template <typename BIT>
    SaberStatus avx_dispatch_with_peephole(const std::vector<DataTensor_in*>& inputs,
                                           std::vector<DataTensor_out*>& outputs,
                                           LstmParam<OpTensor>& param);



};

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
