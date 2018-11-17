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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_GRU_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_GRU_H
#include <vector>
#include "saber/funcs/impl/impl_gru.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/funcs/impl/cuda/cuda_utils.h"
#include "saber/saber_funcs_param.h"
namespace anakin {
namespace saber {

template<DataType OpDtype>
class VenderGru<NV, OpDtype>: public ImplBase <
    NV, OpDtype, GruParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    typedef Tensor<NV> OpTensor;

    VenderGru() : _handle(NULL), _rnnDesc(NULL), _hxDesc(NULL), _cxDesc(NULL), _hyDesc(NULL), \
        _cyDesc(NULL), _wDesc(NULL), _dropoutDesc(NULL), _workspace_size_in_bytes(0),_need_trans(false)  {

    }

    ~VenderGru() {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }

        if (_dropoutDesc) {
            CUDNN_CHECK(cudnnDestroyDropoutDescriptor(_dropoutDesc));
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

    virtual SaberStatus init(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             GruParam<NV>& param,
                             Context<NV>& ctx) override;

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs,
                               std::vector<OpTensor*>& outputs,
                               GruParam<NV>& param,
                               Context<NV>& ctx) override;

    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 GruParam<NV>& param) override;

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
    std::unique_ptr<cudnn::TensorDescriptors<OpDataType>> _xDesc;
    std::unique_ptr<cudnn::TensorDescriptors<OpDataType>> _yDesc;

    int _word_size;
    int _hidden_size;

    OpTensor _inner_weight;
    OpTensor _inner_weight_i2h;
    OpTensor _inner_weight_h2h;
    std::vector<cudnn::ParamsRegion> _inner_weight_region;
    std::vector<cudnn::ParamsRegion> _inner_bias_region;

    bool _need_trans;
    OpTensor _temp_tensor_in;
    OpTensor _temp_tensor_out;

    //! workspace for cudnn
    const size_t _workspace_limit_bytes = 4 * 1024 * 1024;
    size_t _workspace_size_in_bytes;
    OpTensor _workspace_tensor;  // underlying storage

    //! addition flag
    const bool _use_tensor_core = true;

    //! function to transform weights layout to fit cudnn standard
    int get_grnn_params_region(GruParam<NV>& param) ;
    void set_grnn_params_region(GruParam<NV>& param);
    void trans_akweights_2_cudnnweights(GruParam<NV>& param);

    SeqSortedseqTranseUtil _seq_utils;
};

} // namespace saber
} // namespace anakin
#endif // ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_GRU_H
