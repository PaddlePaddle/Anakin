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
#include "saber/funcs/impl/cuda/cuda_utils.h"

namespace anakin {

namespace saber {
struct ParamsRegion {

    ParamsRegion(): _offset(NULL), _size(0) {};
    ParamsRegion(void* offset, size_t size): _offset(offset), _size(size) {}
    ~ParamsRegion() {}
    ParamsRegion(const ParamsRegion& right): _offset(right._offset), _size(right._size) {};

    ParamsRegion& operator=(const ParamsRegion& right) {
        _offset = right._offset;
        _size = right._size;
        return *this;
    }
    bool operator==(const ParamsRegion& right) {
        bool comp_eq = true;
        comp_eq = comp_eq && (_offset == right._offset);
        comp_eq = comp_eq && (_size == right._size);
        return  comp_eq;
    }

    void* _offset;
    size_t _size;
};

template <DataType OpDtype>
class VenderLstm<NV, OpDtype>: \
    public ImplBase <NV,OpDtype,
    LstmParam<NV> >{

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    typedef Tensor<NV> OpTensor;

    VenderLstm()
        : _handle(NULL), _rnn_desc(NULL), _hx_desc(NULL), _cx_desc(NULL), _hy_desc(NULL), \
          _cy_desc(NULL), _w_desc(NULL), _dropout_desc(NULL), _workspace_size_in_bytes(0),_need_trans(false) {}

    ~VenderLstm() {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }

        if (_dropout_desc) {
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

    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             LstmParam<NV>& lstm_param, Context<NV>& ctx) ;

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               LstmParam<NV>& lstm_param, Context<NV>& ctx);


    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 LstmParam<NV>& param);

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
    std::unique_ptr<cudnn::TensorDescriptors<OpDataType>> _x_desc;
    std::unique_ptr<cudnn::TensorDescriptors<OpDataType>> _y_desc;


    OpTensor _inner_weight;
    std::vector<ParamsRegion> _inner_weight_region;
    std::vector<ParamsRegion> _inner_bias_region;
    bool _need_trans;


    int _word_size;
    int _hidden_size;
    bool _is_init_weights = false;
    const int _cudnn_lstm_weights_layernum = 8;

    //! workspace for cudnn
    const size_t _workspace_limit_bytes = 4 * 1024 * 1024;
    size_t _workspace_size_in_bytes;
    OpTensor _workspace_tensor;  // underlying storage

    //! addition flag
    const bool _use_tensor_core = true;

    //! function to transform weights layout to fit cudnn standard
    int get_lstm_params_region(LstmParam<NV>& param);
    void set_lstm_params_region(LstmParam<NV>& param, int wordSize);
    SeqSortedseqTranseUtil _seq_utils;
    OpTensor _temp_tensor_in;
    OpTensor _temp_tensor_out;

};



} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_LSTM_H