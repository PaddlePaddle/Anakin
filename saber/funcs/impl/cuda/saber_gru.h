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
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
namespace anakin {

namespace saber {

template <typename Dtype>
void trans_map2in_cfunc(const Dtype*  input, Dtype* output, int word_size,int seq_sum, cudaStream_t stream,int *dev_map_vec);

template <typename Dtype>
void trans_map2out_cfunc(const Dtype*  input, Dtype* output, int word_size,int seq_sum, cudaStream_t stream,int *dev_map_vec);

class SeqSortedseqTranseUtil {

public:

    SeqSortedseqTranseUtil(bool is_reverse = false, bool is_bi = false)
        : _is_reverse(is_reverse),
          _is_bi(is_bi),
            _dev_map_vec(nullptr),
          _dev_map_vec_length(0)

    {};

    ~SeqSortedseqTranseUtil() {
        if (_dev_map_vec != nullptr) {
            CUDA_CHECK(cudaFree(_dev_map_vec));
        }
    };

    void print_vec(float* in, int size, const char* perfix) {
        for (int i = 0; i < size; i++) {
            printf("[%s] %d = %f\n", perfix, i, in[i]);
        }
    }
    template <typename Dtype>
    void seq_2_sorted_seq(const Dtype*  input, Dtype* output, int word_size, cudaStream_t stream) {
        int seq_sum = _map_vec.size();
        trans_map2out_cfunc(input,output,word_size,seq_sum,stream,_dev_map_vec);
    }
    template <typename Dtype>
    void hidden_2_sorted_hidden(const Dtype*  input, Dtype* output, int hidden_size) {
        //        _map_vec.resize(word_sum);
        int batch_size = _length_index.size();
        //        std::cout << "word_sum = " << word_sum << std::endl;

        for (int ori_word_id = 0; ori_word_id < batch_size; ++ori_word_id) {
            //can param
            int word_start = ori_word_id * hidden_size;
            int maped_id = _length_index[ori_word_id];
            int maped_start = maped_id * hidden_size;

            for (int word_vec_offset = 0; word_vec_offset < hidden_size; ++word_vec_offset) {
                //                std::cout<<maped_start + word_vec_offset<<" --> "<<word_start + word_vec_offset<<" , = "<<input[maped_start + word_vec_offset]<<std::endl;

                output[word_start + word_vec_offset] = input[maped_start + word_vec_offset];

            }
        }
    }
    template <typename Dtype>
    void sorted_seq_2_seq(const Dtype* input, Dtype* output, int hidden_size, cudaStream_t stream) {
        int seq_sum = _map_vec.size();
        trans_map2in_cfunc(input,output,hidden_size,seq_sum,stream,_dev_map_vec);
    }

    bool get_sorted_map(std::vector<int>& offset_vec,
                        std::vector<int>& emit_offset_vec, int& emit_length, cudaStream_t stream_id) {
        int batch_size = offset_vec.size() - 1;
        int word_sum = offset_vec[offset_vec.size() - 1];
        std::vector<int>length_vec(batch_size);
        _length_index.resize(batch_size);

        if (batch_size == 1) {
            emit_length = offset_vec[1] - offset_vec[0];
            emit_offset_vec.resize(emit_length + 1);

            for (int i = 0; i <= emit_length; i++) {
                emit_offset_vec[i] = i;
            }

            return false;
        }

        int max_len = 0;

        for (int i = 0; i < offset_vec.size() - 1; ++i) {
            int len = offset_vec[i + 1] - offset_vec[i];
            max_len = max_len > len ? max_len : len;
            length_vec[i] = len;
            _length_index[i] = i;
        }

        emit_length = max_len;

        if (max_len == 1) {
            emit_offset_vec.push_back(0);
            emit_offset_vec.push_back(emit_length * batch_size);
            return false;
        }

        std::sort(_length_index.begin(), _length_index.end(), [&length_vec](int i1, int i2) {
            return length_vec[i1] > length_vec[i2];
        });

        emit_offset_vec.resize(max_len + 1);
        _map_vec.resize(word_sum);

        if (word_sum > _dev_map_vec_length) {
            if (_dev_map_vec != nullptr) {
                CUDA_CHECK(cudaFree(_dev_map_vec));
            }

            CUDA_CHECK(cudaMalloc(&_dev_map_vec, sizeof(int)*word_sum));
            _dev_map_vec_length = word_sum;
        }

        int target_word_id = 0;
        std::vector<int> length_vec_cnt = length_vec;

        for (int word_id_in_seq = 0; word_id_in_seq < max_len; word_id_in_seq++) {
            emit_offset_vec[word_id_in_seq] = target_word_id;

            for (int batch_id = 0; batch_id < batch_size; batch_id++) {
                int old_batch_id = _length_index[batch_id];

                if (length_vec_cnt[old_batch_id] > 0) {
                    int inner_word_id_in_seq = word_id_in_seq;

                    if (_is_reverse) {
                        inner_word_id_in_seq = length_vec[old_batch_id] - 1 - word_id_in_seq;
                    }

                    int old_word_id = offset_vec[old_batch_id] + inner_word_id_in_seq;
                    _map_vec[old_word_id] = target_word_id;
                    length_vec_cnt[old_batch_id]--;
                    target_word_id++;
                } else {

                    break;
                }
            }
        }


        CUDA_CHECK(cudaMemcpyAsync(_dev_map_vec, _map_vec.data(), sizeof(int)*word_sum,
                                   cudaMemcpyHostToDevice, stream_id));

        emit_offset_vec[max_len] = word_sum;
        return true;
    }

private:

    //    std::vector<int> _length_vec;
    std::vector<int> _length_index;
    std::vector<int> _map_vec;

    int* _dev_map_vec;
    int _dev_map_vec_length;
    bool _is_reverse;
    bool _is_bi;

};

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class SaberGru<NV, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>: \
    public ImplBase <
    Tensor<NV, inDtype, LayOutType_in>, \
    Tensor<NV, outDtype, LayOutType_out>, \
    Tensor<NV, OpDtype, LayOutType_op>, \
    GruParam<Tensor<NV, OpDtype, LayOutType_op> >> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberGru() {}
    ~SaberGru() {
        if (_cublas_handle != NULL) {
            CUBLAS_CHECK(cublasDestroy(_cublas_handle));
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs, \
                             std::vector<DataTensor_out*>& outputs, \
                             GruParam <OpTensor>& gru_param, Context<NV>& ctx) {

        this->_ctx = ctx;
        CUBLAS_CHECK(cublasCreate(&_cublas_handle));
        CUBLAS_CHECK(cublasSetStream(_cublas_handle, this->_ctx.get_compute_stream()));
        if(gru_param.init_hidden()!= nullptr){
            CHECK_EQ(1,0)<<"not support init_hidden now";
        }
        if (gru_param.formula == GRU_ORIGIN) {
            _hidden_size = gru_param.bias()->valid_size() / 3;

            int weights_bias_size = _hidden_size * 3;
            int weights_h2h_size = _hidden_size * _hidden_size * 3;
            int weights_i2h_size = gru_param.weight()->valid_size() - weights_h2h_size;

            _word_size = weights_i2h_size / _hidden_size / 3;
            _weights_i2h.try_expand_size(weights_i2h_size);
            _weights_h2h.try_expand_size(weights_h2h_size);
            _weights_bias.try_expand_size(weights_bias_size);

            int size_data_type = sizeof(InDataType);

            CUDA_CHECK(cudaMemcpy(_weights_i2h.mutable_data(), gru_param.weight()->data(),
                                  size_data_type * weights_i2h_size
                                  , cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(_weights_h2h.mutable_data(), gru_param.weight()->data() + weights_i2h_size,
                                  size_data_type * weights_h2h_size, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(_weights_bias.mutable_data(), gru_param.bias()->data(),
                                  size_data_type * weights_bias_size
                                  , cudaMemcpyDeviceToDevice));
            _seq_util = SeqSortedseqTranseUtil(gru_param.is_reverse);

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

        int batch_size = inputs[0]->get_seq_offset().size() - 1;
        int sequence = inputs[0]->num();
        _gemm_wx = saber_find_fast_sass_gemm(false, false, sequence * batch_size, 3 * _hidden_size,
                                             _word_size);
        _gemm_wh_2 = saber_find_fast_sass_gemm(false, false, batch_size, 2 * _hidden_size, _hidden_size);

        _gemm_wh_o = saber_find_fast_sass_gemm(false, false, batch_size, 1 * _hidden_size, _hidden_size);
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
    Tensor<NV, inDtype, LayOutType_in> _temp_WHR;

    Tensor<NV, inDtype, LayOutType_in> _temp_zero;

    Tensor<NV, AK_INT32, LayOutType_in> _temp_vector_offset;
    Tensor<X86, AK_INT32, LayOutType_in> _temp_map_host;
    Tensor<NV, AK_INT32, LayOutType_in> _temp_map_dev;
    SeqSortedseqTranseUtil _seq_util;
    int _word_size;
    int _hidden_size;

    OpTensor _weights_i2h;
    OpTensor _weights_h2h;
    OpTensor _weights_bias;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wx;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wh_2;

    std::function<void(const int, const int, const int,
                       const float, const float*, const float,
                       const float*, float*, cudaStream_t)> _gemm_wh_o;

    typedef std::function<OpDataType(OpDataType)> ActFunction;

    void seq2hw(std::vector<DataTensor_out*> outputs, std::vector<DataTensor_in*> inputs,
                GruParam<OpTensor>& param, int hidden_size, void* real_temp_out);
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