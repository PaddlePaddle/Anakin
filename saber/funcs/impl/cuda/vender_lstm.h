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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_LSTM_H

#include "saber/funcs/impl/impl_lstm.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/funcs/funcs_utils.h"
#include "saber/funcs/debug.h"
#include "saber/funcs/impl/cuda/base/cuda_c/cuda_utils.h"
#include "cuda_fp16.h"

namespace anakin {

namespace saber {
    struct ParamsRegion {

        ParamsRegion():_offset(NULL), _size(0){};
        ParamsRegion(void *offset, size_t size):_offset(offset), _size(size){}
        ~ParamsRegion(){}
        ParamsRegion(const ParamsRegion &right): _offset(right._offset),_size(right._size){};

        ParamsRegion &operator=(const ParamsRegion &right) {
            _offset = right._offset;
            _size = right._size;
            return *this;
        }
        bool operator==(const ParamsRegion &right) {
            bool comp_eq = true;
            comp_eq = comp_eq && (_offset == right._offset);
            comp_eq = comp_eq && (_size == right._size);
            return  comp_eq;
        }

        void * _offset;
        size_t _size;
    };

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class VenderLstm<NV, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>: \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, \
        Tensor<NV, outDtype, LayOutType_out>, \
        Tensor<NV, OpDtype, LayOutType_op>, \
        LstmParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor;
    typedef Tensor<NV, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor::Dtype DataDtype;
    typedef typename OpTensor::Dtype Op_dtype;

    VenderLstm()
        : _handle(NULL), _rnn_desc(NULL), _hx_desc(NULL), _cx_desc(NULL), _hy_desc(NULL), \
        _cy_desc(NULL), _w_desc(NULL), _dropout_desc(NULL), _workspace_size_in_bytes(0) {}

    ~VenderLstm() {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
        if(_dropout_desc){
            CUDNN_CHECK(cudnnDestroyDropoutDescriptor(_dropout_desc));
        }
        if (_rnn_desc) {
            CUDNN_CHECK(cudnnDestroyRNNDescriptor(_rnn_desc));
        }

        if (_hx_desc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_hx_desc));
        }
        if (_cx_desc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_cx_desc));
        }
        if (_hy_desc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_hy_desc));
        }
        if (_cy_desc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_cy_desc));
        }

        if (_w_desc) {
            CUDNN_CHECK(cudnnDestroyFilterDescriptor(_w_desc));
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor*>& inputs,
                         std::vector<OutDataTensor*>& outputs,
                         LstmParam<OpTensor> &lstm_param, Context<NV> &ctx) {
        if (lstm_param.with_peephole) {
            return SaberInvalidValue;
        }

        _workspace_size_in_bytes = 0;

        this->_ctx = &ctx;
        // ---- get cuda resources ----

        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();

        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

        // ---- create cudnn Descs ----
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&_dropout_desc));
        CUDNN_CHECK(cudnnCreateRNNDescriptor(&_rnn_desc));

        cudnn::createTensorDesc<DataDtype>(&_hx_desc);
        cudnn::createTensorDesc<DataDtype>(&_cx_desc);
        cudnn::createTensorDesc<DataDtype>(&_hy_desc);
        cudnn::createTensorDesc<DataDtype>(&_cy_desc);

        cudnn::createFilterDesc<Op_dtype>(&_w_desc);
        //cudnnSetDropoutDescriptor(_dropout_desc,
        //                   _handle,
        //                   lstm_param.dropout_param,
        //                   NULL,
        //                   0,
        //                   0);
       //_dropout_desc = NULL;
       _hidden_size = lstm_param.bias()->valid_size() / 4 / lstm_param.num_layers;
       int weights_h2h_size = _hidden_size * _hidden_size * 4 * lstm_param.num_layers;
       int weights_i2h_size = lstm_param.weight()->valid_size() - weights_h2h_size;
       _word_size = weights_i2h_size / (4 * _hidden_size);
       
       cudnn::setRNNDesc<DataDtype>(&_rnn_desc, _handle, _hidden_size,
                                     lstm_param.num_layers, _dropout_desc, lstm_param.num_direction, CUDNN_LSTM);
       _x_desc.reset(new cudnn::TensorDescriptors<DataDtype>(
                        1,
       {1/*batch_size*/, _word_size, 1},
       {_word_size, 1, 1}));
        size_t  weights_size = 0;
        CUDNN_CHECK(cudnnGetRNNParamsSize(
                        _handle,
                        _rnn_desc,
                        _x_desc->descs()[0],
                        & weights_size,
                        cudnn::cudnnTypeWrapper<DataDtype>::type));

        const int dims[] = {
            static_cast<int>( weights_size / sizeof(Op_dtype)),
            1,
            1
        };
        CUDNN_CHECK(cudnnSetFilterNdDescriptor(
                        _w_desc, cudnn::cudnnTypeWrapper<Op_dtype >::type, CUDNN_TENSOR_NCHW, 3, dims));
        /**
         * in_weights is tensor of char not the opdata
         */
        Shape weight_tensor_shape(1, 1, 1,  weights_size / sizeof(Op_dtype));
        _inner_weight.reshape(weight_tensor_shape);
        int sum_size_of_w = get_lstm_params_region(lstm_param);
        CHECK_EQ(sum_size_of_w,  weights_size) << "Compute param sum length must equal to that api get." ;
        set_lstm_params_region(lstm_param, _word_size);
        

       SeqSortedseqTranseUtil seq_utils(lstm_param.is_reverse, lstm_param.num_direction == 2 ? true : false);
       _seq_utils = seq_utils;
        return create(inputs, outputs, lstm_param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor*>& inputs,
                           std::vector<OutDataTensor*>& outputs,
                           LstmParam<OpTensor> &lstm_param, Context<NV> &ctx);


    virtual SaberStatus dispatch(const std::vector<DataTensor*>& inputs,
                             std::vector<OutDataTensor*>& outputs,
                             LstmParam<OpTensor> &param);

private:

    cudnnHandle_t _handle;

//! choose for lstm or lstm or rnn

    cudnnDropoutDescriptor_t _dropout_desc;

    cudnnRNNDescriptor_t _rnn_desc;

//! gate desc have to be valid
    cudnnTensorDescriptor_t _hx_desc;
    cudnnTensorDescriptor_t _cx_desc;
    cudnnTensorDescriptor_t _hy_desc;
    cudnnTensorDescriptor_t _cy_desc;
    cudnnFilterDescriptor_t _w_desc;

//! input and output descs
    std::unique_ptr<cudnn::TensorDescriptors<DataDtype>> _x_desc;
    std::unique_ptr<cudnn::TensorDescriptors<DataDtype>> _y_desc;


    OpTensor _inner_weight;
    std::vector<ParamsRegion> _inner_weight_region;
    std::vector<ParamsRegion> _inner_bias_region;

    int _word_size;
    int _hidden_size;
    bool _is_init_weights = false;
    const int _cudnn_lstm_weights_layernum = 8;

//! workspace for cudnn
    const size_t _workspace_limit_bytes = 4 * 1024 * 1024;
    size_t _workspace_size_in_bytes;
    Tensor<NV, AK_INT8, NCHW> _workspace_tensor;  // underlying storage

//! addition flag
    const bool _use_tensor_core = true;

//! function to transform weights layout to fit cudnn standard
    int get_lstm_params_region(LstmParam<OpTensor> &param);
    void set_lstm_params_region(LstmParam<OpTensor> &param, int wordSize);
    SeqSortedseqTranseUtil _seq_utils;
    DataTensor _temp_tensor_in;
    DataTensor _temp_tensor_out;

};

template class VenderLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_LSTM_H
