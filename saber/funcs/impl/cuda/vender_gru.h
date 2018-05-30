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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_GRU_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_GRU_H

#include "saber/funcs/impl/impl_gru.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/funcs/funcs_utils.h"
#include "saber/funcs/debug.h"
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
            _size=right._size;
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
class VenderGru<NV, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>: \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, \
        Tensor<NV, outDtype, LayOutType_out>, \
        Tensor<NV, OpDtype, LayOutType_op>, \
        GruParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor;
    typedef Tensor<NV, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor::Dtype DataDtype;
    typedef typename OpTensor::Dtype Op_dtype;

    VenderGru()
        : _handle(NULL), _rnnDesc(NULL), _hxDesc(NULL), _cxDesc(NULL), _hyDesc(NULL), \
        _cyDesc(NULL), _wDesc(NULL), _dropoutDesc(NULL), _workspace_size_in_bytes(0) {}

    ~VenderGru() {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
        if (_rnnDesc) {
            CUDNN_CHECK(cudnnDestroyRNNDescriptor(_rnnDesc));
        }

        if (_hxDesc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_hxDesc));
        }
        if (_cxDesc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_cxDesc));
        }
        if (_hyDesc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_hyDesc));
        }
        if (_cyDesc) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_cyDesc));
        }

        if (_wDesc) {
            CUDNN_CHECK(cudnnDestroyFilterDescriptor(_wDesc));
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor*>& inputs,
                         std::vector<OutDataTensor*>& outputs,
                         GruParam<OpTensor> &gru_param, Context<NV> &ctx) {

        _workspace_size_in_bytes = 0;

        this->_ctx = ctx;
        // ---- get cuda resources ----

        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();

        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));



        // ---- create cudnn Descs ----
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&_dropoutDesc));
        CUDNN_CHECK(cudnnCreateRNNDescriptor(&_rnnDesc));

        cudnn::createTensorDesc<DataDtype>(&_hxDesc);
        cudnn::createTensorDesc<DataDtype>(&_cxDesc);
        cudnn::createTensorDesc<DataDtype>(&_hyDesc);
        cudnn::createTensorDesc<DataDtype>(&_cyDesc);

        cudnn::createFilterDesc<Op_dtype>(&_wDesc);

        if(_is_init_weights==false){
            _hidden_size = gru_param.bias()->valid_size() / 3;

            int weights_bias_size = _hidden_size * 3;
            int weights_h2h_size = _hidden_size * _hidden_size * 3;
            int weights_i2h_size = gru_param.weight()->valid_size() - weights_h2h_size;
            _word_size = weights_i2h_size / _hidden_size / 3;

            Tensor<X86, OpDtype, LayOutType_op> inner_weights_before_host;
            Shape weights_shape(1,1,1,gru_param.weight()->valid_size());
            inner_weights_before_host.re_alloc(weights_shape);
            inner_weights_before_host.copy_from(*gru_param.weight());

            const Op_dtype* weights_i2h=inner_weights_before_host.data();
            const Op_dtype* weights_h2h=weights_i2h+weights_i2h_size;




            int temp_size=_hidden_size>_word_size?_hidden_size*_hidden_size:_word_size*_hidden_size;
            Shape temp_tensor_shape(1,1,1,temp_size);
            Tensor<X86, OpDtype, LayOutType_op> temp_tensor;
            temp_tensor.re_alloc(temp_tensor_shape);
            extract_matrix_from_matrix_in_leddim(weights_i2h,temp_tensor.mutable_data(),0,weights_i2h_size,_hidden_size*3,_hidden_size);
            write_tensorfile(temp_tensor,"temp_tensor");
        }



        return create(inputs, outputs, gru_param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor*>& inputs,
                           std::vector<OutDataTensor*>& outputs,
                           GruParam<OpTensor> &gru_param, Context<NV> &ctx);


    virtual SaberStatus dispatch(const std::vector<DataTensor*>& inputs,
                             std::vector<OutDataTensor*>& outputs,
                             GruParam<OpTensor> &param);

private:

    cudnnHandle_t _handle;

//! choose for lstm or gru or rnn

    cudnnDropoutDescriptor_t _dropoutDesc;

    cudnnRNNDescriptor_t _rnnDesc;

//! gate desc have to be valid
    cudnnTensorDescriptor_t _hxDesc;
    cudnnTensorDescriptor_t _cxDesc;
    cudnnTensorDescriptor_t _hyDesc;
    cudnnTensorDescriptor_t _cyDesc;
    cudnnFilterDescriptor_t _wDesc;
    const int _cudnn_gru_weights_layernum = 6;

//! input and output descs
    std::unique_ptr<cudnn::TensorDescriptors<DataDtype>> _xDesc;
    std::unique_ptr<cudnn::TensorDescriptors<DataDtype>> _yDesc;

    int _word_size;
    int _hidden_size;
    bool _is_init_weights=false;

    OpTensor _inner_weight;
    std::vector<ParamsRegion> _inner_weight_region;
    std::vector<ParamsRegion> _inner_bias_region;


//! workspace for cudnn
    const size_t _workspace_limit_bytes = 64 * 1024 * 1024;
    size_t _workspace_size_in_bytes;
    Tensor<NV, AK_INT8, NCHW> _workspace_tensor;  // underlying storage

//! addition flag
    const bool _use_tensor_core = true;

//! function to transform weights layout to fit cudnn standard
    int get_grnn_params_region(GruParam<OpTensor> &param) ;
    void set_grnn_params_region(GruParam<OpTensor> &param, int wordSize);
    void hw2seq(std::vector<DataTensor*> inputs, GruParam<OpTensor>& param, int word_size,
            DataTensor &sequence, DataTensor &out_sequence, Context<NV>& ctx);
    void seq2hw(std::vector<DataTensor*> outputs, std::vector<DataTensor*> inputs,
            GruParam<OpTensor>& param, int hidden_size, DataTensor &sequence,
            Context<NV>& ctx);
};

template class VenderGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_GRU_H
