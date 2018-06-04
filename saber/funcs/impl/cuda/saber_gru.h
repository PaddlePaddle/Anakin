/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

namespace anakin {

namespace saber {

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
class SaberGru<NV, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, \
        Tensor<NV, outDtype, LayOutType_out>, \
        Tensor<NV, OpDtype, LayOutType_op>, \
        GruParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberGru() {}
    ~SaberGru() {
        if (_cublas_handle != nullptr) {
            CUBLAS_CHECK(cublasDestroy(_cublas_handle));
        }
    }
    
    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, \
        GruParam <OpTensor>& gru_param, Context<NV>& ctx) {

        this->_ctx = ctx;
        CUBLAS_CHECK(cublasCreate(&_cublas_handle));
        CUBLAS_CHECK(cublasSetStream(_cublas_handle, this->_ctx.get_compute_stream()));
        if(gru_param.formula==GRU_ORIGIN) {
            _hidden_size = gru_param.bias()->valid_size() / 3;

            int weights_bias_size = _hidden_size * 3;
            int weights_h2h_size = _hidden_size * _hidden_size * 3;
            int weights_i2h_size = gru_param.weight()->valid_size() - weights_h2h_size;

            _word_size = weights_i2h_size / _hidden_size / 3;
            _weights_i2h.try_expand_size(weights_i2h_size);
            _weights_h2h.try_expand_size(weights_h2h_size);
            _weights_bias.try_expand_size(weights_bias_size);

            int size_data_type = sizeof(InDataType);
//            memcpy(_weights_i2h.mutable_data(), gru_param.weight()->data(),
//                   size_data_type * weights_i2h_size);
//            memcpy(_weights_h2h.mutable_data(), gru_param.weight()->data() + weights_i2h_size,
//                   size_data_type * weights_h2h_size);
//            memcpy(_weights_bias.mutable_data(), gru_param.bias()->data(),
//                   size_data_type * weights_bias_size);

            CUDA_CHECK(cudaMemcpy(_weights_i2h.mutable_data(), gru_param.weight()->data(), size_data_type * weights_i2h_size
                    ,cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(_weights_h2h.mutable_data(), gru_param.weight()->data() + weights_i2h_size,
                                  size_data_type * weights_h2h_size,cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(_weights_bias.mutable_data(), gru_param.bias()->data(), size_data_type * weights_bias_size
                    ,cudaMemcpyDeviceToDevice));
        }
        return create(inputs, outputs, gru_param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, \
        GruParam<OpTensor>& gru_param, Context<NV>& ctx) {

        if (!(ctx == this->_ctx)) {
            if (_cublas_handle != NULL) {
                CUBLAS_CHECK(cublasDestroy(_cublas_handle));
            }
            this->_ctx = ctx;

            cudaStream_t cuda_stream;
            cuda_stream = ctx.get_compute_stream();
            CUBLAS_CHECK(cublasCreate(&_cublas_handle));
            CUBLAS_CHECK(cublasSetStream(_cublas_handle, cuda_stream));
        }

        return SaberSuccess;
    }


    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 GruParam <OpTensor>& param);

private:
    cublasHandle_t  _cublas_handle;
/**
 * for hw2seq
 */
    Tensor<NV, inDtype, LayOutType_in> _temp_tensor_in;
    Tensor<NV, inDtype, LayOutType_in> _temp_tensor_out;
    Tensor<NV, inDtype, LayOutType_in> _temp_WX;
    Tensor<NV, inDtype, LayOutType_in> _temp_WH;

    Tensor<NV, AK_INT32, LayOutType_in> _temp_vector_offset;
    Tensor<X86, AK_INT32, LayOutType_in> _temp_map_host;
    Tensor<NV, AK_INT32, LayOutType_in> _temp_map_dev;

    int _word_size;
    int _hidden_size;

    OpTensor _weights_i2h;
    OpTensor _weights_h2h;
    OpTensor _weights_bias;

    void seq2hw(std::vector<DataTensor_out*> outputs, std::vector<DataTensor_in*> inputs,
                GruParam<OpTensor>& param, int hidden_size,void* real_temp_out);
/**
 * dim2 input to seq,batch,wordsize
 * @param inputs
 * @param param
 * @param word_size
 * @param sequence
 * @param out_sequence
 * @param ctx
 * @return sequence length
 */
    const InDataType* hw2seq(std::vector<DataTensor_in*> inputs, GruParam<OpTensor>& param,
                            int word_size, int hiddensize, int& sequence_len);

    SaberStatus gru_cudnn(const std::vector<DataTensor_in*> inputs,
                          std::vector<DataTensor_out*> outputs,
                          GruParam<OpTensor>& param);
};



} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_GRU_H