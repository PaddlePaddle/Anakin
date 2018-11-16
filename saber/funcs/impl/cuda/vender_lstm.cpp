#include "saber/funcs/impl/cuda/vender_lstm.h"

#include "saber/core/tensor_op.h"

namespace anakin {
namespace saber {

template <>
void VenderLstm<NV, AK_FLOAT>::\
set_lstm_params_region(LstmParam<NV>& param, int word_size) {
    int hidden_size = param.bias()->valid_size() / 4;

    if (param.with_peephole) {
        hidden_size = param.bias()->valid_size() / 7;
    }

    int bias_size_per_layer = 4 * hidden_size;
    int weight_size_per_layer = 4 * hidden_size * (word_size + hidden_size);
    int wx_stride = hidden_size;
    int wh_stride = hidden_size;
    cudaStream_t cuda_stream;
    cuda_stream = this->_ctx->get_compute_stream();

    for (int layer_id = 0; layer_id < param.num_layers; layer_id++) {
        const OpDataType* w_ptr = static_cast<const OpDataType*>(param.weight()->data()) + layer_id * weight_size_per_layer;
        const OpDataType* w_xi = w_ptr;
        const OpDataType* w_xf = w_ptr + wx_stride;
        const OpDataType* w_xc = w_ptr + 2 * wx_stride;
        const OpDataType* w_xo = w_ptr + 3 * wx_stride;

        const OpDataType* w_ptr_inner  = w_ptr + 4 * wx_stride * word_size;
        const OpDataType* w_hi = w_ptr_inner;
        const OpDataType* w_hf = w_ptr_inner + 1 * wh_stride;
        const OpDataType* w_hc = w_ptr_inner + 2 * wh_stride;
        const OpDataType* w_ho = w_ptr_inner + 3 * wh_stride;

        const OpDataType* b_i = nullptr;
        const OpDataType* b_f = nullptr;
        const OpDataType* b_c = nullptr;
        const OpDataType* b_o = nullptr;

        if (param.bias() != nullptr) {
            b_i = static_cast<const OpDataType*>(param.bias()->data()) + layer_id * bias_size_per_layer;
            b_f = b_i + hidden_size;
            b_c = b_f + hidden_size;
            b_o = b_c + hidden_size;
        }

        const OpDataType* cudnnW[] = {w_xi, w_xf, w_xc, w_xo, w_hi, w_hf, w_hc, w_ho};
        const OpDataType* cudnnB[] = {b_i, b_f, b_c, b_o, nullptr, nullptr, nullptr, nullptr};

        for (int i = 0; i < _cudnn_lstm_weights_layernum; i++) {
            ParamsRegion& region = _inner_weight_region[i];
            get_sub_tensor<OpDataType>(cudnnW[i], (OpDataType*) region._offset,
                                     region._size / (sizeof(OpDataType) * hidden_size), hidden_size, 4 * hidden_size, cuda_stream);
        }

        for (int i = 0; i < _cudnn_lstm_weights_layernum; i++) {
            ParamsRegion& region_b = _inner_bias_region[i];

            if (cudnnB[i] != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync((void*)(region_b._offset), (void*)cudnnB[i],
                                           region_b._size,
                                           cudaMemcpyDeviceToDevice, cuda_stream));
            } else {
                CUDA_CHECK(cudaMemsetAsync((void*)(region_b._offset), 0, region_b._size, cuda_stream));
            }
        }
    }
}

template <>
int VenderLstm<NV, AK_FLOAT>::\
get_lstm_params_region(LstmParam<NV>& param) {
    int sum_size_of_weights_and_bias = 0;
    cudnnFilterDescriptor_t region_desc_handle = nullptr;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&region_desc_handle));
    /**
     * lstm in rnn has 8 bias layer
     */
    int region_count_of_layer = _cudnn_lstm_weights_layernum;
    _inner_weight_region.clear();
    _inner_bias_region.clear();

    for (int layer = 0; layer < param.num_layers; layer++) {
        for (int region = 0; region < region_count_of_layer; region++) {
            for (int trigger = 0; trigger < 2; trigger++) {
                void* offset = nullptr;

                if (trigger == 0) { /*  weights */
                    CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(_handle,
                                _rnn_desc,
                                layer,
                                _x_desc->descs()[0],
                                _w_desc,
                                _inner_weight.mutable_data(),            /* nullptr */
                                region,             /* linLayerID */
                                region_desc_handle, /* linLayerMatDesc */
                                &offset));
                } else { /* bias */
                    CUDNN_CHECK(cudnnGetRNNLinLayerBiasParams(_handle,
                                _rnn_desc,
                                layer,
                                _x_desc->descs()[0],
                                _w_desc,
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
                auto regionp = ParamsRegion{offset, size};

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
SaberStatus VenderLstm<NV, AK_FLOAT>::\
create(const std::vector<OpTensor*>& inputs,
       std::vector<OpTensor*>& outputs,
       LstmParam<NV>& param, Context<NV>& ctx) {

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
SaberStatus VenderLstm<NV, AK_FLOAT>::\
        init(const std::vector<OpTensor*>& inputs,
                 std::vector<OpTensor*>& outputs,
                 LstmParam<NV>& lstm_param, Context<NV>& ctx) {
    if (lstm_param.with_peephole) {
        return SaberInvalidValue;
    }

    _workspace_size_in_bytes = 0;
    _workspace_tensor.set_dtype(AK_INT8);
    this->_ctx = &ctx;
    // ---- get cuda resources ----

    cudaStream_t cuda_stream;
    cuda_stream = ctx.get_compute_stream();

    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

    // ---- create cudnn Descs ----
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&_dropout_desc));
    CUDNN_CHECK(cudnnCreateRNNDescriptor(&_rnn_desc));

    cudnn::createTensorDesc<OpDataType>(&_hx_desc);
    cudnn::createTensorDesc<OpDataType>(&_cx_desc);
    cudnn::createTensorDesc<OpDataType>(&_hy_desc);
    cudnn::createTensorDesc<OpDataType>(&_cy_desc);

    cudnn::createFilterDesc<OpDataType>(&_w_desc);
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

    cudnn::setRNNDesc<OpDataType>(&_rnn_desc, _handle, _hidden_size,
                                  lstm_param.num_layers, _dropout_desc, lstm_param.num_direction, CUDNN_LSTM);

    _x_desc.reset(new cudnn::TensorDescriptors<OpDataType>(
            1,
            {{1/*batch_size*/, _word_size, 1}},
            {{_word_size, 1, 1}}));

    size_t  weights_size = 0;
    CUDNN_CHECK(cudnnGetRNNParamsSize(
            _handle,
            _rnn_desc,
            _x_desc->descs()[0],
            & weights_size,
            cudnn::cudnnTypeWrapper<OpDataType>::type));

    const int dims[] = {
            static_cast<int>(weights_size / sizeof(OpDataType)),
            1,
            1
    };
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(
            _w_desc, cudnn::cudnnTypeWrapper<OpDataType >::type, CUDNN_TENSOR_NCHW, 3, dims));
    /**
     * in_weights is tensor of char not the opdata
     */
    Shape weight_tensor_shape({1, 1, 1,  weights_size / sizeof(OpDataType)});
    _inner_weight.reshape(weight_tensor_shape);
    int sum_size_of_w = get_lstm_params_region(lstm_param);
    CHECK_EQ(sum_size_of_w,  weights_size) << "Compute param sum length must equal to that api get." ;
    set_lstm_params_region(lstm_param, _word_size);

    _seq_utils = SeqSortedseqTranseUtil(lstm_param.is_reverse,
                                        lstm_param.num_direction == 2 ? true : false);
    return create(inputs, outputs, lstm_param, ctx);
}


template<>
SaberStatus VenderLstm<NV, AK_FLOAT>::\
dispatch(const std::vector<OpTensor*>& inputs,
         std::vector<OpTensor*>& outputs,
         LstmParam<NV>& param) {
    CHECK_GE(inputs.size(), 1) << "lstm input vec size must >=1";
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    auto seq_offset_vec=inputs[0]->get_seq_offset();
            CHECK_GT(seq_offset_vec.size(),0);
    auto seq_offset = seq_offset_vec[seq_offset_vec.size()-1];
            CHECK_GT(seq_offset.size(),0);

    _need_trans=_seq_utils.get_sorted_map(seq_offset, this->_ctx->get_compute_stream());

    int max_seq_len = _seq_utils.get_emit_offset_vec().size() - 1;
    int batch_size = seq_offset.size() - 1;//H

    size_t state_size;
    auto offset_after_sort = _seq_utils.get_emit_offset_vec();

    std::vector<std::vector<int>> xdim(max_seq_len);
    std::vector<std::vector<int>> xstride(max_seq_len);

    for (int i = 0; i < max_seq_len; i++) {
        int length=offset_after_sort[i+1]-offset_after_sort[i];
        xdim[i] = {length, _word_size, 1};
        xstride[i] = {_word_size, 1, 1};
    }

    _x_desc.reset(new cudnn::TensorDescriptors<OpDataType>(max_seq_len, xdim, xstride));

    std::vector<std::vector<int>> ydim(max_seq_len);
    std::vector<std::vector<int>> ystride(max_seq_len);

    for (int i = 0; i < max_seq_len; i++) {
        int length=offset_after_sort[i+1]-offset_after_sort[i];
        ydim[i] = {length, _hidden_size * param.num_direction, 1};
        ystride[i] = {_hidden_size * param.num_direction, 1, 1};
    }

    _y_desc.reset(new cudnn::TensorDescriptors<OpDataType>(max_seq_len, ydim, ystride));

    Shape in_dim = inputs[0]->valid_shape();
    Shape in_stride = inputs[0]->get_stride();

    Shape out_dim = outputs[0]->valid_shape();
    Shape out_stride = outputs[0]->get_stride();

    int dim[] = {param.num_layers * param.num_direction, batch_size, _hidden_size};
    int stride[] = {batch_size * _hidden_size, _hidden_size, 1};

    cudnn::setTensorNdDesc<OpDataType >(&_hx_desc,
                                        3, dim, stride);
    cudnn::setTensorNdDesc<OpDataType >(&_cx_desc,
                                        3, dim, stride);
    cudnn::setTensorNdDesc<OpDataType >(&_hy_desc,
                                        3, dim, stride);
    cudnn::setTensorNdDesc<OpDataType >(&_cy_desc,
                                        3, dim, stride);
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
            _handle,
            _rnn_desc,
            max_seq_len,
            _x_desc->descs(),
            &_workspace_size_in_bytes));

    _workspace_tensor.reshape(Shape({1, 1, 1, _workspace_size_in_bytes}));

    int input_channel = inputs[0]->channel();
    const OpDataType* in_data = static_cast<const OpDataType*>(inputs[0]->data());
    OpDataType* out_data = static_cast<OpDataType*>(outputs[0]->mutable_data());
    const OpDataType* in_hidden_data = nullptr;

    if (inputs.size() == 2) {
        in_hidden_data = static_cast<const OpDataType*>(inputs[1]->data());
    }


    if (_need_trans) {
        _temp_tensor_in.reshape(inputs[0]->valid_shape());
        _temp_tensor_out.reshape(outputs[0]->valid_shape());
        OpDataType* temp_in_data = static_cast<OpDataType*>(_temp_tensor_in.mutable_data());
        OpDataType* temp_out_data = static_cast<OpDataType*>(_temp_tensor_out.mutable_data());
        _seq_utils.seq_2_sorted_seq(in_data, temp_in_data, _word_size,
                                    this->_ctx->get_compute_stream());
        CUDNN_CHECK(cudnnRNNForwardInference(_handle,
                                             _rnn_desc,
                                             _x_desc->sizes(),//sequence
                                             _x_desc->descs(),
                                             temp_in_data,
                                             _hx_desc,
                                             in_hidden_data, // hidden state of the network will be initialized to zero
                                             _cx_desc,
                                             nullptr, //the initial cell state of the network will be initialized to zero
                                             _w_desc,
                                             _inner_weight.data(),
                                             _y_desc->descs(),
                                             temp_out_data,  // Output GPU-raw-ptr
                                             _hy_desc,
                                             nullptr, // the final hidden state of the network will not be saved
                                             _cy_desc,
                                             nullptr, //  the final cell state of the network will be not be saved
                                             _workspace_tensor.mutable_data(),
                                             _workspace_size_in_bytes));
        _seq_utils.sorted_seq_2_seq(temp_out_data, out_data, _hidden_size,
                                    this->_ctx->get_compute_stream());

    } else {
        CUDNN_CHECK(cudnnRNNForwardInference(_handle,
                                             _rnn_desc,
                                             _x_desc->sizes(),
                                             _x_desc->descs(),
                                             in_data,
                                             _hx_desc,
                                             in_hidden_data, // hidden state of the network will be initialized to zero
                                             _cx_desc,
                                             nullptr, //the initial cell state of the network will be initialized to zero
                                             _w_desc,
                                             _inner_weight.data(),
                                             _y_desc->descs(),
                                             out_data,  // Output GPU-raw-ptr
                                             _hy_desc,
                                             nullptr, // the final hidden state of the network will not be saved
                                             _cy_desc,
                                             nullptr, //  the final cell state of the network will be not be saved
                                             _workspace_tensor.mutable_data(),
                                             _workspace_size_in_bytes));
    }

    return SaberSuccess;
}

template class VenderLstm<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderLstm, LstmParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderLstm, LstmParam, NV, AK_INT8);

}
}