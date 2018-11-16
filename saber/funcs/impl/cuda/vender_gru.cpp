#include "saber/funcs/impl/cuda/vender_gru.h"
#include "cuda_fp16.h"
#include "tensor_op.h"
namespace anakin {
namespace saber {

template <>
void VenderGru<NV, AK_FLOAT>::\
set_grnn_params_region(GruParam<NV>& param) {
    const OpDataType* w_i2h_ptr = static_cast<const OpDataType*>
                                  (_inner_weight_i2h.data());                /*inpute weights*/
    const OpDataType* w_h2h_ptr = static_cast<const OpDataType*>(_inner_weight_h2h.data());
    CHECK_NOTNULL(w_i2h_ptr) << "weights can`t be null";
    CHECK_NOTNULL(w_h2h_ptr) << "weights can`t be null";
    const OpDataType* i2h = w_i2h_ptr;                                   /* new memory gate */
    const OpDataType* i2h_r = w_i2h_ptr + 1 * _word_size * _hidden_size;     /* reset gate */
    const OpDataType* i2h_z = w_i2h_ptr + 2 * _word_size * _hidden_size;     /* update gate */
    const OpDataType* h2h = w_h2h_ptr;                             /* new memory gate */
    const OpDataType* h2h_r = w_h2h_ptr + 1 * _hidden_size * _hidden_size; /* reset gate */
    const OpDataType* h2h_z = w_h2h_ptr + 2 * _hidden_size * _hidden_size; /* update gate */

    const OpDataType* h = nullptr;
    const OpDataType* h_r = nullptr;
    const OpDataType* h_z = nullptr;

    if (param.bias() != nullptr) {
        h = static_cast<const OpDataType* >(param.bias()->data());
        h_r = h + 1 * _hidden_size;
        h_z = h + 2 * _hidden_size;
    }

    const OpDataType* cudnnW[] = {i2h_r, i2h_z, i2h, h2h_r, h2h_z, h2h};
    const OpDataType* cudnnB[] = {h_r, h_z, h, nullptr, nullptr, nullptr};

    for (int i = 0; i < _cudnn_gru_weights_layernum; i++) {
        cudnn::ParamsRegion& region = _inner_weight_region[i];
        CUDA_CHECK(cudaMemcpy((void*)(region._offset), (void*)cudnnW[i],
                              region._size,
                              cudaMemcpyDeviceToDevice));
    }

    for (int i = 0; i < _cudnn_gru_weights_layernum; i++) {
        cudnn::ParamsRegion& region_b = _inner_bias_region[i];

        if (cudnnB[i] != nullptr) {
            CUDA_CHECK(cudaMemcpy((void*)(region_b._offset), (void*)cudnnB[i],
                                  region_b._size,
                                  cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemset((void*)(region_b._offset), 0, region_b._size));
        }
    }
}

template <>
int VenderGru<NV, AK_FLOAT>::\
get_grnn_params_region(GruParam<NV>& param) {
    int sum_size_of_weights_and_bias = 0;
    cudnnFilterDescriptor_t region_desc_handle = nullptr;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&region_desc_handle));
    /**
     * gru in rnn has 6 bias layer
     */
    int region_count_of_layer = _cudnn_gru_weights_layernum;


    for (int layer = 0; layer < param.num_layers; layer++) {
        for (int region = 0; region < region_count_of_layer; region++) {
            for (int trigger = 0; trigger < 2; trigger++) {
                void* offset = nullptr;

                if (trigger == 0) { /*  weights */
                    CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(_handle,
                                _rnnDesc,
                                layer,
                                _xDesc->descs()[0],
                                _wDesc,
                                _inner_weight.mutable_data(),            /* nullptr */
                                region,             /* linLayerID */
                                region_desc_handle, /* linLayerMatDesc */
                                &offset));
                } else { /* bias */
                    CUDNN_CHECK(cudnnGetRNNLinLayerBiasParams(_handle,
                                _rnnDesc,
                                layer,
                                _xDesc->descs()[0],
                                _wDesc,
                                _inner_weight.mutable_data(),
                                region,
                                region_desc_handle,   /* linLayerBiasDesc */
                                &offset));
                }

                int dims[] = {1, 1, 1};
                cudnnDataType_t data_type;
                cudnnTensorFormat_t tensor_format;
                int nbDims;
                CUDNN_CHECK(cudnnGetFilterNdDescriptor(region_desc_handle,           /* filterDesc */
                                                       sizeof(dims) / sizeof(dims[0]), /* nbDimsRequested */
                                                       &data_type,
                                                       &tensor_format,
                                                       &nbDims,
                                                       dims));                         /* filterDimA[] */
                size_t size = dims[0] * dims[1] * dims[2] * sizeof(OpDataType);
                sum_size_of_weights_and_bias += size;
                auto regionp = cudnn::ParamsRegion{offset, size};

                if (trigger == 0) { /* weights */
                    _inner_weight_region.push_back(regionp);
                } else { /* bias */
                    _inner_bias_region.push_back(regionp);
                }
            }
        }
    }

    return sum_size_of_weights_and_bias;
}

template<>
SaberStatus VenderGru<NV, AK_FLOAT>::\
create(const std::vector<Tensor<NV>*>& inputs,
       std::vector<Tensor<NV>*>& outputs,
       GruParam<NV>& param, Context<NV>& ctx) {

    if (!(&ctx == this->_ctx)) {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }

        this->_ctx = &ctx;
        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();

        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));
    }


    return SaberSuccess;
}

template<>
void VenderGru<NV, AK_FLOAT>::trans_akweights_2_cudnnweights(GruParam<NV>& param) {
    int weights_i2h_size = 3*_hidden_size * _word_size;
    Tensor<X86> weight_i2h_host;
    Tensor<X86> weight_i2h_host_target;
    Tensor<X86> weight_trans_workspace;
    Shape weight_i2h_host_shape({1, 1, 1, _hidden_size* _word_size * 3});
    weight_i2h_host.re_alloc(weight_i2h_host_shape, AK_FLOAT);
    weight_i2h_host_target.re_alloc(weight_i2h_host_shape, AK_FLOAT);
    weight_trans_workspace.re_alloc(weight_i2h_host_shape, AK_FLOAT);
    _inner_weight_i2h.re_alloc(weight_i2h_host_shape, AK_FLOAT);

    CUDA_CHECK(cudaMemcpyAsync(weight_i2h_host.mutable_data(), param.weight()->data(),
                               sizeof(OpDataType)*_hidden_size * _word_size * 3, cudaMemcpyDeviceToHost,
                               this->_ctx->get_compute_stream()));
    CUDA_CHECK(cudaDeviceSynchronize());

    OpDataType* rz_temp_tensor_ptr = static_cast<OpDataType*>(weight_i2h_host_target.mutable_data());
    OpDataType* rz_weights_tensor_ptr = static_cast<OpDataType*>(weight_i2h_host.mutable_data());

    for (int row = 0; row < _word_size; row++) {
        for (int block = 0; block < 3; block++) {
            int block_offset = block * _hidden_size;

            for (int cow = 0; cow < _hidden_size; cow++) {
                rz_temp_tensor_ptr[block * _word_size * _hidden_size + row * _hidden_size + cow] =
                    rz_weights_tensor_ptr[row * (3 * _hidden_size) + cow + block_offset];
            }
        }
    }

    weight_trans_workspace.copy_from(weight_i2h_host_target);
    const OpDataType* rz_weight_trans_workspace_ptr = static_cast<const OpDataType*>
            (weight_trans_workspace.data());

    for (int i = 0; i < 3; i++) {
        utils::transpose(rz_weight_trans_workspace_ptr + i * _hidden_size * _word_size, _word_size,
                         _hidden_size,
                         rz_temp_tensor_ptr + i * _hidden_size * _word_size);
    }

    _inner_weight_i2h.copy_from(weight_i2h_host_target);

    Tensor<X86> weight_h2h_host;
    Tensor<X86> weight_h2h_host_target;
    Shape weight_h2h_host_shape({1, 1, 1, _hidden_size* _hidden_size * 3});
    _inner_weight_h2h.re_alloc(weight_h2h_host_shape, AK_FLOAT);
    weight_h2h_host.re_alloc(weight_h2h_host_shape, AK_FLOAT);
    weight_h2h_host_target.re_alloc(weight_h2h_host_shape, AK_FLOAT);
    weight_trans_workspace.re_alloc(weight_h2h_host_shape, AK_FLOAT);

    CUDA_CHECK(cudaMemcpyAsync(weight_h2h_host.mutable_data(),
                               static_cast<const OpDataType*>(param.weight()->data()) + weights_i2h_size,
                               sizeof(OpDataType)*_hidden_size * _hidden_size * 3, cudaMemcpyDeviceToHost,
                               this->_ctx->get_compute_stream()));
    CUDA_CHECK(cudaDeviceSynchronize());

    memcpy(weight_h2h_host_target.mutable_data(), weight_h2h_host.data(),
           _hidden_size * _hidden_size * sizeof(OpDataType));
    rz_temp_tensor_ptr = static_cast<OpDataType*>(weight_h2h_host_target.mutable_data()) + _hidden_size
                         * _hidden_size;
    rz_weights_tensor_ptr = static_cast<OpDataType*>(weight_h2h_host.mutable_data()) + _hidden_size *
                            _hidden_size;

    for (int row = 0; row < _hidden_size; row++) {
        for (int block = 0; block < 2; block++) {
            int block_offset = block * _hidden_size;

            for (int cow = 0; cow < _hidden_size; cow++) {
                rz_temp_tensor_ptr[block * _hidden_size * _hidden_size + row * _hidden_size + cow] =
                    rz_weights_tensor_ptr[row * (2 * _hidden_size) + cow + block_offset];
            }
        }
    }

    weight_trans_workspace.copy_from(weight_h2h_host_target);

    rz_weight_trans_workspace_ptr = static_cast<const OpDataType*>(weight_trans_workspace.data());
    rz_temp_tensor_ptr = static_cast<OpDataType*>(weight_h2h_host_target.mutable_data());

    for (int i = 0; i < 3; i++) {
        utils::transpose(rz_weight_trans_workspace_ptr + i * _hidden_size * _hidden_size, _hidden_size,
                         _hidden_size,
                         rz_temp_tensor_ptr + i * _hidden_size * _hidden_size);
    }

    _inner_weight_h2h.copy_from(weight_h2h_host_target);
    CUDA_CHECK(cudaDeviceSynchronize());

}


template<>
SaberStatus VenderGru<NV, AK_FLOAT>::
init(const std::vector<Tensor<NV>*>& inputs,
     std::vector<Tensor<NV>*>& outputs,
     GruParam<NV>& param,
     Context<NV>& ctx) {

    _hidden_size = param.bias()->valid_size() / 3;
    int weights_bias_size = _hidden_size * 3;
    int weights_h2h_size = _hidden_size * _hidden_size * 3;
    int weights_i2h_size = param.weight()->valid_size() - weights_h2h_size;
    _word_size = weights_i2h_size / _hidden_size / 3;
    _workspace_size_in_bytes = 0;

    this->_ctx = &ctx;
    // ---- get cuda resources ----

    cudaStream_t cuda_stream;
    cuda_stream = ctx.get_compute_stream();

    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));


    // ---- create cudnn Descs ----
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&_dropoutDesc));
    CUDNN_CHECK(cudnnCreateRNNDescriptor(&_rnnDesc));
    cudnn::setRNNDesc<OpDataType>(&_rnnDesc, _handle, _hidden_size,
                                  param.num_layers, _dropoutDesc, param.num_direction, CUDNN_GRU);

    cudnn::createTensorDesc<OpDataType>(&_hxDesc);
    cudnn::createTensorDesc<OpDataType>(&_cxDesc);
    cudnn::createTensorDesc<OpDataType>(&_hyDesc);
    cudnn::createTensorDesc<OpDataType>(&_cyDesc);
    cudnn::createFilterDesc<OpDataType>(&_wDesc);
    _workspace_tensor.set_dtype(AK_INT8);

    _xDesc.reset(new cudnn::TensorDescriptors<OpDataType>(
                     1,
    {{1/*batch_size*/, _word_size, 1}},
    {{_word_size, 1, 1}}));

    size_t weightsSize = 999;
    CUDNN_CHECK(cudnnGetRNNParamsSize(
                    _handle,
                    _rnnDesc,
                    _xDesc->descs()[0],
                    &weightsSize,
                    CUDNN_DATA_FLOAT));
    const int dims[] = {
        static_cast<int>(weightsSize) / sizeof(OpDataType),
        1,
        1
    };

    Shape weight_tensor_shape({1, 1, 1, weightsSize / sizeof(OpDataType)});
    _inner_weight.re_alloc(weight_tensor_shape, AK_FLOAT);

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(
                    _wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dims));

    /**
     * in_weights is tensor of char not the opdata
     */

    trans_akweights_2_cudnnweights(param);

    int sum_size_of_w = get_grnn_params_region(param);
    CHECK_EQ(sum_size_of_w, weightsSize) << "Compute param sum length must equal to that api get." ;

    set_grnn_params_region(param);

    _seq_utils=SeqSortedseqTranseUtil();
    return create(inputs, outputs, param, ctx);
};

template<>
SaberStatus VenderGru<NV, AK_FLOAT>::\
dispatch(const std::vector<Tensor<NV>*>& inputs,
         std::vector<Tensor<NV>*>& outputs,
         GruParam<NV>& param) {
    CHECK_GE(inputs.size(), 1) << "gru input vec size must >=1";
    const OpDataType* in_data = static_cast<const OpDataType*>(inputs[0]->data());
    OpDataType* out_data = static_cast<OpDataType*>(outputs[0]->mutable_data());
    const OpDataType* in_hidden_data = nullptr;
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    std::vector<std::vector<int>> offset_vec = inputs[0]->get_seq_offset();
    std::vector<int> offset = offset_vec[offset_vec.size() - 1];
    int batch_size = offset.size() - 1;

    _need_trans=_seq_utils.get_sorted_map(offset,this->_ctx->get_compute_stream());
    int max_seq_len = _seq_utils.get_emit_offset_vec().size() - 1;
    auto offset_after_sort = _seq_utils.get_emit_offset_vec();

    std::vector<std::vector<int>> xdim(max_seq_len);
    std::vector<std::vector<int>> xstride(max_seq_len);

    for (int i = 0; i < max_seq_len; i++) {
        int length=offset_after_sort[i+1]-offset_after_sort[i];
        xdim[i] = {length, _word_size, 1};
        xstride[i] = {_word_size, 1, 1};
    }

    _xDesc.reset(new cudnn::TensorDescriptors<OpDataType>(max_seq_len, xdim, xstride));

    std::vector<std::vector<int>> ydim(max_seq_len);
    std::vector<std::vector<int>> ystride(max_seq_len);

    for (int i = 0; i < max_seq_len; i++) {
        int length=offset_after_sort[i+1]-offset_after_sort[i];
        ydim[i] = {length, _hidden_size * param.num_direction, 1};
        ystride[i] = {_hidden_size * param.num_direction, 1, 1};
    }

    _yDesc.reset(new cudnn::TensorDescriptors<OpDataType>(max_seq_len, ydim, ystride));

    int dim[] = {param.num_layers * param.num_direction, batch_size, _hidden_size};
    int stride[] = {batch_size * _hidden_size, _hidden_size, 1};

    cudnn::setTensorNdDesc<OpDataType >(&_hxDesc,
                                        3, dim, stride);
    cudnn::setTensorNdDesc<OpDataType >(&_cxDesc,
                                        3, dim, stride);
    cudnn::setTensorNdDesc<OpDataType >(&_hyDesc,
                                        3, dim, stride);
    cudnn::setTensorNdDesc<OpDataType >(&_cyDesc,
                                        3, dim, stride);

    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
            _handle,
            _rnnDesc,
            max_seq_len,
            _xDesc->descs(),
            &_workspace_size_in_bytes));

    utils::try_expand_tensor(_workspace_tensor, _workspace_size_in_bytes);


    if (inputs.size() == 2) {
        in_hidden_data = (const OpDataType*)inputs[1]->data();
    }
    if(_need_trans) {

        utils::try_expand_tensor(_temp_tensor_in, inputs[0]->valid_shape());
        utils::try_expand_tensor(_temp_tensor_out, outputs[0]->valid_shape());
        OpDataType *temp_in_data = static_cast<OpDataType *>(_temp_tensor_in.mutable_data());
        OpDataType *temp_out_data = static_cast<OpDataType *>(_temp_tensor_out.mutable_data());
        _seq_utils.seq_2_sorted_seq(in_data, temp_in_data, _word_size, this->_ctx->get_compute_stream());

        CUDNN_CHECK(cudnnRNNForwardInference(_handle,
                                             _rnnDesc,
                                             _xDesc->sizes(),
                                             _xDesc->descs(),
                                             temp_in_data,
                                             _hxDesc,
                                             in_hidden_data, // hidden state of the network will be initialized to zero
                                             _cxDesc,
                                             nullptr, //the initial cell state of the network will be initialized to zero
                                             _wDesc,
                                             _inner_weight.data(),
                                             _yDesc->descs(),
                                             temp_out_data,  // Output GPU-raw-ptr
                                             _hyDesc,
                                             nullptr, // the final hidden state of the network will not be saved
                                             _cyDesc,
                                             nullptr, //  the final cell state of the network will be not be saved
                                             _workspace_tensor.mutable_data(),
                                             _workspace_size_in_bytes));
        _seq_utils.sorted_seq_2_seq(temp_out_data, out_data, _hidden_size, this->_ctx->get_compute_stream());
    }else{
        CUDNN_CHECK(cudnnRNNForwardInference(_handle,
                                             _rnnDesc,
                                             _xDesc->sizes(),
                                             _xDesc->descs(),
                                             in_data,
                                             _hxDesc,
                                             in_hidden_data, // hidden state of the network will be initialized to zero
                                             _cxDesc,
                                             nullptr, //the initial cell state of the network will be initialized to zero
                                             _wDesc,
                                             _inner_weight.data(),
                                             _yDesc->descs(),
                                             out_data,  // Output GPU-raw-ptr
                                             _hyDesc,
                                             nullptr, // the final hidden state of the network will not be saved
                                             _cyDesc,
                                             nullptr, //  the final cell state of the network will be not be saved
                                             _workspace_tensor.mutable_data(),
                                             _workspace_size_in_bytes));
    }

    return SaberSuccess;
}

template class VenderGru<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderGru, GruParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderGru, GruParam, NV, AK_INT8);


}
}
