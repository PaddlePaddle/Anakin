#include "saber/funcs/impl/cuda/vender_gru.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {



template <>
void VenderGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
    seq2hw(std::vector<DataTensor*> outputs, std::vector<DataTensor*> inputs,
            GruParam<OpTensor>& param, int hidden_size, 
            DataTensor& sequence, Context<NV>& ctx) {
    DataTensor* din = inputs[0];
    DataTensor* dout = outputs[0];
    std::vector<int> offset_vec = din->get_seq_offset();
    CHECK_GE(offset_vec.size(), 2) << "offset must >=2" ;
    int batch_size = offset_vec.size() - 1;
    int max_len = 0;
    std::vector<int> length_vec;

    for (int i = 0; i < offset_vec.size() - 1; ++i) {
        int len = offset_vec[i + 1] - offset_vec[i];
        max_len = max_len > len ? max_len : len;
        length_vec.push_back(len);
    }

    const DataDtype* orgin = sequence.data();
    DataDtype* target = dout->mutable_data();

    int count = 0;

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int seq = 0; seq < length_vec[batch]; ++seq) {
            const DataDtype* origin_i = orgin + (seq * batch_size + batch) * hidden_size;
            DataDtype* target_i = target + (offset_vec[batch] + seq) * hidden_size;
            count += hidden_size;
            CUDA_CHECK(cudaMemcpyAsync(target_i, origin_i, sizeof(DataDtype)*hidden_size,
                                       cudaMemcpyDeviceToDevice, ctx.get_data_stream()));
        }
    }

    CHECK_EQ(count, dout->valid_size()) << "output data size should be equal";
    cudaStreamSynchronize(ctx.get_data_stream());
}

template <>
void VenderGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
hw2seq(std::vector<DataTensor*> inputs, GruParam<OpTensor>& param,
        int word_size, DataTensor& sequence, 
        DataTensor& out_sequence, Context<NV>& ctx) {
    DataTensor* din = inputs[0];
    std::vector<int> offset_vec = din->get_seq_offset();
    CHECK_GE(offset_vec.size(), 2) << "offset must >=2" ;
    int batch_size = offset_vec.size() - 1;
    int max_len = 0;
    int hidden_size = param.bias()->valid_size() / 3;
    std::vector<int> length_vec;

    for (int i = 0; i < offset_vec.size() - 1; ++i) {
        int len = offset_vec[i + 1] - offset_vec[i];
        max_len = max_len > len ? max_len : len;
        length_vec.push_back(len);
    }

    Shape seq_shape(1, max_len, batch_size, word_size);
    sequence.re_alloc(seq_shape);

    Shape seq_out_shape(1, max_len, batch_size, hidden_size);
    out_sequence.re_alloc(seq_out_shape);

    if (batch_size == 1) {
        sequence.copy_from(*din);
        return;
    }

    DataDtype* target = sequence.mutable_data();
    const DataDtype* origin = din->data();

    DataTensor zero_tensor;
    Shape zero_shape(1, 1, 1, word_size);
    zero_tensor.re_alloc(zero_shape);

    DataDtype* zero_block = zero_tensor.mutable_data();
    //TODO:set all zero
    CUDA_CHECK(cudaMemset(zero_block, 0, sizeof(DataDtype) * (word_size)));

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int seq = 0; seq < max_len; ++seq) {
            DataDtype* target_i = target + (seq * batch_size + batch) * word_size;

            if (seq < length_vec[batch]) {
                const DataDtype* origin_i = origin + (offset_vec[batch] + seq) * word_size;
                CUDA_CHECK(cudaMemcpyAsync(target_i, origin_i, sizeof(DataDtype)*word_size,
                                           cudaMemcpyDeviceToDevice, ctx.get_data_stream()));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(target_i, zero_block, sizeof(DataDtype)*word_size,
                                           cudaMemcpyDeviceToDevice, ctx.get_data_stream()));
            }
        }
    }

    _xDesc.reset(new cudnn::TensorDescriptors<DataDtype>(
                     max_len,
    {batch_size, word_size, 1},
    {word_size, 1, 1}));

    _yDesc.reset(new cudnn::TensorDescriptors<DataDtype>(
                     max_len,
    {batch_size, hidden_size * param.num_direction, 1},
    {hidden_size  * param.num_direction, 1, 1}));

    size_t new_workspace_size_in_bytes = 0;
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
                    _handle,
                    _rnnDesc,
                    max_len,
                    _xDesc->descs(),
                    &new_workspace_size_in_bytes));

    if (new_workspace_size_in_bytes > _workspace_size_in_bytes) {
        _workspace_size_in_bytes = new_workspace_size_in_bytes;
        _workspace_tensor.re_alloc(Shape(1, 1, 1, _workspace_size_in_bytes));
    }

    int dim[] = {param.num_layers * param.num_direction, batch_size, hidden_size};
    int stride[] = {batch_size * hidden_size, hidden_size, 1};

    cudnn::setTensorNdDesc<DataDtype >(&_hxDesc,
                                       3, dim, stride);
    cudnn::setTensorNdDesc<DataDtype >(&_cxDesc,
                                       3, dim, stride);
    cudnn::setTensorNdDesc<DataDtype >(&_hyDesc,
                                       3, dim, stride);
    cudnn::setTensorNdDesc<DataDtype >(&_cyDesc,
                                       3, dim, stride);

    cudaStreamSynchronize(ctx.get_data_stream());
}

template <>
void VenderGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
set_grnn_params_region(GruParam<OpTensor>& param, int wordSize) {
    int hidden_size = param.bias()->valid_size() / 3;
    const Op_dtype* w_ptr = param.weight()->data();                /*inpute weights*/
    const Op_dtype* i2h = w_ptr;                                   /* new memory gate */
    const Op_dtype* i2h_r = w_ptr + 1 * wordSize * hidden_size;     /* reset gate */
    const Op_dtype* i2h_z = w_ptr + 2 * wordSize * hidden_size;     /* update gate */
    const Op_dtype* w_ptr_inner  = w_ptr + 3 * wordSize * hidden_size;
    const Op_dtype* h2h = w_ptr_inner;                             /* new memory gate */
    const Op_dtype* h2h_r = w_ptr_inner + 1 * hidden_size * hidden_size; /* reset gate */
    const Op_dtype* h2h_z = w_ptr_inner + 2 * hidden_size * hidden_size; /* update gate */

    const Op_dtype* h = nullptr;
    const Op_dtype* h_r = nullptr;
    const Op_dtype* h_z = nullptr;

    if (param.bias() != nullptr) {
        h = param.bias()->data();
        h_r = h + 1 * hidden_size;
        h_z = h + 2 * hidden_size;
    }

    const Op_dtype* cudnnW[] = {i2h_r, i2h_z, i2h, h2h_r, h2h_z, h2h};
    const Op_dtype* cudnnB[] = {h_r, h_z, h, nullptr, nullptr, nullptr};

    for (int i = 0; i < _cudnn_gru_weights_layernum; i++) {
        ParamsRegion& region = _inner_weight_region[i];
        CUDA_CHECK(cudaMemcpy((void*)(region._offset), (void*)cudnnW[i],
                              region._size,
                              cudaMemcpyDeviceToDevice));
    }

    for (int i = 0; i < _cudnn_gru_weights_layernum; i++) {
        ParamsRegion& region_b = _inner_bias_region[i];

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
int VenderGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
get_grnn_params_region(GruParam<OpTensor>& param) {
    int sum_size_of_weights_and_bias = 0;
    cudnnFilterDescriptor_t region_desc_handle = nullptr;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&region_desc_handle));
    /**
     * gru in rnn has 6 bias layer
     */
    int region_count_of_layer = _cudnn_gru_weights_layernum;
    //    LOG(INFO) << "numLayers= " << param.numLayers << ",region_count_of_layer=" << region_count_of_layer;

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
                size_t size = dims[0] * dims[1] * dims[2] * sizeof(Op_dtype);
                //                        LOG(INFO) << "size add  "<<size<<",layer"<<layer<<",region"<<region<<"trigger"<<trigger;
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
SaberStatus VenderGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
create(const std::vector<DataTensor*>& inputs,
       std::vector<OutDataTensor*>& outputs,
       GruParam<OpTensor>& gru_param, Context<NV>& ctx) {

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

    int input_num = inputs[0]->num();
    int input_channel = inputs[0]->channel();
    int input_height = inputs[0]->height();
    int input_width = inputs[0]->width();
    int output_channel = outputs[0]->channel();
    int output_height = outputs[0]->height();
    int output_width = outputs[0]->width();

    int seqLength = input_channel;//C;
    int batchSize = input_height;//H
    size_t stateSize;

    cudnn::setRNNDesc<DataDtype>(&_rnnDesc, _handle, _hidden_size,
                                 gru_param.num_layers, _dropoutDesc, gru_param.num_direction, CUDNN_GRU);

    _xDesc.reset(new cudnn::TensorDescriptors<DataDtype>(
                     seqLength,
    {batchSize, _word_size, 1},
    {_word_size, 1, 1}));

    _yDesc.reset(new cudnn::TensorDescriptors<DataDtype>(
                     seqLength,
    {batchSize, _hidden_size * gru_param.num_direction, 1},
    {_hidden_size  * gru_param.num_direction, 1, 1}));

    Shape in_dim = inputs[0]->shape();
    Shape in_stride = inputs[0]->get_stride();

    Shape out_dim = outputs[0]->shape();
    Shape out_stride = outputs[0]->get_stride();

    int dim[] = {gru_param.num_layers * gru_param.num_direction, batchSize, _hidden_size};
    int stride[] = {batchSize * _hidden_size, _hidden_size, 1};

    cudnn::setTensorNdDesc<DataDtype >(&_hxDesc,
                                       3, dim, stride);
    cudnn::setTensorNdDesc<DataDtype >(&_cxDesc,
                                       3, dim, stride);
    cudnn::setTensorNdDesc<DataDtype >(&_hyDesc,
                                       3, dim, stride);
    cudnn::setTensorNdDesc<DataDtype >(&_cyDesc,
                                       3, dim, stride);

    size_t weightsSize = 0;
    CUDNN_CHECK(cudnnGetRNNParamsSize(
                    _handle,
                    _rnnDesc,
                    _xDesc->descs()[0],
                    &weightsSize,
                    cudnn::cudnnTypeWrapper<DataDtype>::type));

    const int dims[] = {
        static_cast<int>(weightsSize / sizeof(Op_dtype)),
        1,
        1
    };
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(
                    _wDesc, cudnn::cudnnTypeWrapper<Op_dtype >::type, CUDNN_TENSOR_NCHW, 3, dims));
    /**
     * in_weights is tensor of char not the opdata
     */
    Shape weight_tensor_shape(1, 1, 1, weightsSize / sizeof(Op_dtype));
    _inner_weight.re_alloc(weight_tensor_shape);

    int sum_size_of_w = get_grnn_params_region(gru_param);
    CHECK_EQ(sum_size_of_w, weightsSize) << "Compute param sum length must equal to that api get." ;
    set_grnn_params_region(gru_param, _word_size);

    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
                    _handle,
                    _rnnDesc,
                    seqLength,
                    _xDesc->descs(),
                    &_workspace_size_in_bytes));

    _workspace_tensor.re_alloc(Shape(1, 1, 1, _workspace_size_in_bytes));
    return SaberSuccess;
}
template<>
SaberStatus VenderGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::\
dispatch(const std::vector<DataTensor*>& inputs,
        std::vector<OutDataTensor*>& outputs,
        GruParam<OpTensor>& param) {
    CHECK_GE(inputs.size(), 1) << "gru input vec size must >=1";
    int input_channel = inputs[0]->channel();
    const DataDtype* in_data = inputs[0]->data();
    DataDtype* out_data = outputs[0]->mutable_data();
    const DataDtype* in_hidden_data = nullptr;

    if (inputs.size() == 2) {
        in_hidden_data = inputs[1]->data();
    }
    bool isHW2Seq=inputs[0]->get_seq_offset().size()>2;

    if (isHW2Seq) {
        DataTensor temp_tensor_in;
        DataTensor temp_tensor_out;
        hw2seq(inputs, param, _word_size, temp_tensor_in, temp_tensor_out, *_ctx);
        CUDNN_CHECK(cudnnRNNForwardInference(_handle,
                                             _rnnDesc,
                                             _xDesc->sizes(),//sequence
                                             _xDesc->descs(),
                                             temp_tensor_in.data(),
                                             _hxDesc,
                                             in_hidden_data, // hidden state of the network will be initialized to zero
                                             _cxDesc,
                                             nullptr, //the initial cell state of the network will be initialized to zero
                                             _wDesc,
                                             _inner_weight.data(),
                                             _yDesc->descs(),
                                             temp_tensor_out.mutable_data(),  // Output GPU-raw-ptr
                                             _hyDesc,
                                             nullptr, // the final hidden state of the network will not be saved
                                             _cyDesc,
                                             nullptr, //  the final cell state of the network will be not be saved
                                             _workspace_tensor.mutable_data(),
                                             _workspace_size_in_bytes));

        seq2hw(outputs, inputs, param, _hidden_size, temp_tensor_out, *_ctx);
        outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    } else {
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


}
}
