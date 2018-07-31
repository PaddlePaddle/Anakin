#include "saber/funcs/impl/cuda/vender_lstm.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <>
void VenderLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
set_lstm_params_region(LstmParam<OpTensor>& param, int word_size) {
    int hidden_size = param.bias()->valid_size() / 4;
    if (param.with_peephole) {
        hidden_size = param.bias()->valid_size() / 7;
    }
    int bias_size_per_layer = 4 * hidden_size;
    int weight_size_per_layer = 4 * hidden_size * (word_size + hidden_size);
    int wx_stride = word_size * hidden_size;
    int wh_stride = hidden_size * hidden_size;
    cudaStream_t cuda_stream;
    cuda_stream = this->_ctx->get_compute_stream();
    for (int layer_id = 0; layer_id < param.num_layers; layer_id++) {
        const Op_dtype* w_ptr = param.weight()->data() + layer_id * weight_size_per_layer;
        const Op_dtype* w_xi = w_ptr;                
        const Op_dtype* w_xf = w_ptr + wx_stride;     
        const Op_dtype* w_xc = w_ptr + 2 * wx_stride; 
        const Op_dtype* w_xo = w_ptr + 3 * wx_stride; 

        const Op_dtype* w_ptr_inner  = w_ptr + 4 * wx_stride;
        const Op_dtype* w_hi = w_ptr_inner;                 
        const Op_dtype* w_hf = w_ptr_inner + 1 * wh_stride; 
        const Op_dtype* w_hc = w_ptr_inner + 2 * wh_stride; 
        const Op_dtype* w_ho = w_ptr_inner + 3 * wh_stride; 

        const Op_dtype* b_i = nullptr;
        const Op_dtype* b_f = nullptr;
        const Op_dtype* b_c = nullptr;
        const Op_dtype* b_o = nullptr;

        if (param.bias() != nullptr) {
            b_i = param.bias()->data() + layer_id * bias_size_per_layer;
            b_f = b_i + hidden_size;
            b_c = b_f + hidden_size;
            b_o = b_c + hidden_size;
        }

        const Op_dtype* cudnnW[] = {w_xi, w_xf, w_xc, w_xo, w_hi, w_hf, w_hc, w_ho}; 
        const Op_dtype* cudnnB[] = {b_i, b_f, b_c, b_o, nullptr, nullptr, nullptr, nullptr};

        for (int i = 0; i < _cudnn_lstm_weights_layernum; i++) {
            ParamsRegion& region = _inner_weight_region[i];
            CUDA_CHECK(cudaMemcpyAsync((void*)(region._offset), (void*)cudnnW[i],
                                  region._size,
                                  cudaMemcpyDeviceToDevice, cuda_stream));
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
int VenderLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
get_lstm_params_region(LstmParam<OpTensor>& param) {
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
                size_t size = dims[0] * dims[1] * dims[2] * sizeof(Op_dtype);
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
SaberStatus VenderLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
create(const std::vector<DataTensor*>& inputs,
       std::vector<OutDataTensor*>& outputs,
       LstmParam<OpTensor>& lstm_param, Context<NV>& ctx) {

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

    auto seq_offset = inputs[0]->get_seq_offset();
    if (seq_offset.size() == 0) {
        return SaberSuccess;
    }
    _seq_utils.get_sorted_map(seq_offset, ctx.get_compute_stream());

    int max_seq_len = _seq_utils.get_emit_offset_vec().size() - 1;
    int batch_size = seq_offset.size() - 1;//H

    size_t state_size;
    auto offset_after_sort = _seq_utils.get_emit_offset_vec();

    _x_desc.reset(new cudnn::TensorDescriptors<DataDtype>(
                     offset_after_sort,
    {batch_size, _word_size, 1},
    {_word_size, 1, 1}));

    _y_desc.reset(new cudnn::TensorDescriptors<DataDtype>(
                     offset_after_sort,
    {batch_size, _hidden_size * lstm_param.num_direction, 1},
    {_hidden_size  * lstm_param.num_direction, 1, 1}));

    Shape in_dim = inputs[0]->valid_shape();
    Shape in_stride = inputs[0]->get_stride();

    Shape out_dim = outputs[0]->valid_shape();
    Shape out_stride = outputs[0]->get_stride();

    int dim[] = {lstm_param.num_layers * lstm_param.num_direction, batch_size, _hidden_size};
    int stride[] = {batch_size * _hidden_size, _hidden_size, 1};

    cudnn::setTensorNdDesc<DataDtype >(&_hx_desc,
                                       3, dim, stride);
    cudnn::setTensorNdDesc<DataDtype >(&_cx_desc,
                                       3, dim, stride);
    cudnn::setTensorNdDesc<DataDtype >(&_hy_desc,
                                       3, dim, stride);
    cudnn::setTensorNdDesc<DataDtype >(&_cy_desc,
                                       3, dim, stride);
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
                    _handle,
                    _rnn_desc,
                    max_seq_len,
                    _x_desc->descs(),
                    &_workspace_size_in_bytes));

    _workspace_tensor.reshape(Shape(1, 1, 1, _workspace_size_in_bytes));
    return SaberSuccess;
}
template<>
SaberStatus VenderLstm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
dispatch(const std::vector<DataTensor*>& inputs,
        std::vector<OutDataTensor*>& outputs,
        LstmParam<OpTensor>& param) {
    CHECK_GE(inputs.size(), 1) << "lstm input vec size must >=1";
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    int input_channel = inputs[0]->channel();
    const DataDtype* in_data = inputs[0]->data();
    DataDtype* out_data = outputs[0]->mutable_data();
    const DataDtype* in_hidden_data = nullptr;

    if (inputs.size() == 2) {
        in_hidden_data = inputs[1]->data();
    }
    bool isHW2Seq = inputs[0]->get_seq_offset().size() > 2;

    if (isHW2Seq) {
        _temp_tensor_in.reshape(inputs[0]->valid_shape());
        _temp_tensor_out.reshape(outputs[0]->valid_shape());
        auto temp_in_data = _temp_tensor_in.mutable_data();
        auto temp_out_data = _temp_tensor_out.mutable_data();
        _seq_utils.seq_2_sorted_seq(inputs[0]->data(), temp_in_data, _hidden_size, this->_ctx->get_compute_stream());
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
        _seq_utils.sorted_seq_2_seq(temp_out_data, out_data, _hidden_size, this->_ctx->get_compute_stream());

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


}
}
