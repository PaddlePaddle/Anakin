#include "saber/funcs/impl/cuda/saber_attension_lstm.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/debug.h"
#include "cuda_utils.h"
#include "sass_funcs.h"
namespace anakin {

namespace saber {
static void gemm(cublasHandle_t handle,
                 const bool TransA, const bool TransB,
                 int m, int n, int k, const float alpha,
                 const float* a, const float* b,
                 const float beta, float* c) {
    //    cout << "(" << m << "," << n << "," << k << ")" << endl;
    int lda = (!TransA/* == CblasNoTrans*/) ? k : m;
    int ldb = (!TransB/* == CblasNoTrans*/) ? n : k;
    cublasOperation_t cuTransA =
        (!TransA/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (!TransB/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasSgemm(handle, cuTransA, cuTransB, m, n, k, &alpha, b, ldb, a, lda, &beta, c, n);
};
/*one block compute one sequence*/
/*use share memory to reduce*/
template <typename Dtype>
__global__ void sequence_softmax(const Dtype* in_data, const int* seq_offset, const int seq_num,
                                 Dtype* out_data) {
    int t_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (t_id >= seq_num) {
        return;
    }

    int start_id = seq_offset[t_id];
    int end_id = seq_offset[t_id + 1];
    Dtype max_data = -1e32;
    //Dtype max_data = -FLT_MAX;
    Dtype sum = 0;

    for (int i = start_id; i < end_id; i++) {
        max_data = in_data[i] > max_data ? in_data[i] : max_data;
    }

    for (int i = start_id; i < end_id; i++) {
        sum +=  expf(in_data[i] - max_data);
    }

    for (int i = start_id; i < end_id; i++) {
        out_data[i] =  expf(in_data[i] - max_data) / sum;
    }
}
template <typename Dtype>
__global__ void relu(const Dtype* in_data, Dtype* out_data, int count) {
    int t_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (t_id >= count) {
        return;
    }

    out_data[t_id] = in_data[t_id] > 0 ? in_data[t_id] : 0;
}

template <typename Dtype>
__global__ void bias_relu(const Dtype* in_data, const Dtype* bias_data, const int count,
                          const int bias_size, Dtype* out_data) {
    int t_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (t_id >= count) {
        return;
    }

    int id = t_id % bias_size;
    Dtype data = in_data[t_id] + bias_data[id];
    out_data[t_id] = data > 0 ? data : 0;
}

template <typename Dtype>
__global__ void sequence_pool(const Dtype* in_data, const Dtype* scale, const int* seq_offset,
                              const int seq_num, const int total_num, const int dim, Dtype* out_data) {
    int t_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (t_id >= seq_num * dim) {
        return;
    }

    int dim_id = t_id % dim;
    int seq_id = t_id / dim;
    int start_id = seq_offset[seq_id];
    int end_id = seq_offset[seq_id + 1];
    Dtype sum = 0;
    const Dtype* in = in_data + dim_id + start_id * dim;

    for (int i = 0; i < end_id - start_id; i++) {
        sum += in[0] * scale[i + start_id];
        in += dim;
    }

    out_data[t_id] = sum;
}
template<typename Dtype>
__device__ Dtype sigmoid(Dtype in) {
    Dtype out = Dtype(1.0) / (1 + exp(-in));
    return out;
}

template<typename Dtype>
__device__ Dtype tanh(Dtype in) {
    //Dtype out = (exp(in)- exp(-in)) / (exp(in) + exp(-in));
    //Dtype out = 1 - 2.f / (expf(2*in) + 1);
    //Dtype out = 1 - 2.f / (expf(2*in) + 1);
    Dtype a = expf(in);
    Dtype b = expf(-in);
    return (a - b) / (a + b);
}

template <typename Dtype>
__global__ void lstm_bias_and_act(const Dtype* in_data, const Dtype* bias_data, Dtype* out_data,
                                  Dtype* cell_data, int batch_size, int hidden_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= batch_size * hidden_size) {
        return;
    }

    int dim_id = tid % hidden_size;
    int batch_id  = tid / hidden_size;
    int offset = batch_id * hidden_size + dim_id;
    const Dtype* tmp_in = in_data + batch_id * 4 * hidden_size + dim_id;
    Dtype* tmp_cell = cell_data + offset;
    const Dtype* tmp_bias = bias_data + dim_id;
    Dtype ct  = tanh(tmp_in[2 * hidden_size] + tmp_bias[2 * hidden_size]);
    Dtype ig  = sigmoid(tmp_in[0 * hidden_size] + tmp_bias[0 * hidden_size]);
    Dtype fg  = sigmoid(tmp_in[1 * hidden_size] + tmp_bias[1 * hidden_size]);
    Dtype og  = sigmoid(tmp_in[3 * hidden_size] + tmp_bias[3 * hidden_size]);
    tmp_cell[0] = ig * ct  + fg * tmp_cell[0];
    out_data[offset] = og * tanh(tmp_cell[0]);
}

template <typename Dtype>
__global__ void sequence_bias_relu(const Dtype* in_data, const Dtype* seq_bias,
                                   const Dtype* bias_data, const int* seq_id, const int num, const int dim,
                                   Dtype* out_data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= dim * num) {
        return;
    }

    int dim_id = tid % dim;
    int word_id = tid / dim;
    int cur_seq_id = seq_id[word_id];
    Dtype data  = in_data[tid] + seq_bias[cur_seq_id * dim + dim_id] + bias_data[dim_id];
    //printf("%d, in:%f,  seq_bias:%f, bias:%f\n", tid, in_data[tid], seq_bias[cur_seq_id * dim + dim_id], bias_data[dim_id]);
    out_data[tid] = data > 0 ? data : 0;
}

template <typename Dtype>
__global__ void lstm_result_to_sequence(const Dtype* in_data, const int* seq_id_map,
                                        const int* offset, const int seq_num,
                                        const int word_num, const int hidden_size, Dtype* out_data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= hidden_size * word_num) {
        return;
    }

    int dim_id = tid % hidden_size;
    int word_id = tid / hidden_size;
    int seq_id = seq_id_map[word_id];
    int word_id_in_seq = word_id - offset[seq_id];
    out_data[tid] = in_data[(word_id_in_seq * seq_num  + seq_id) * hidden_size + dim_id];
}

template<>
SaberStatus SaberAttensionLstm<NV, AK_FLOAT>:: create(const std::vector<OpTensor*>& inputs, \
               std::vector<OpTensor*>& outputs, \
               AttensionLstmParam<NV>& attension_lstm_param, Context<NV>& ctx) {

    if(inputs[0]->get_seq_offset().size()>0) {
        int batch_size = inputs[0]->get_seq_offset()[0].size() - 1;
        int sequence = inputs[0]->num();

        _gemm_wx = saber_find_fast_sass_gemm(false, false,
                                             sequence, 4 * _hidden_size, _word_size);
        _gemm_wh = saber_find_fast_sass_gemm(false, false, batch_size, 4 * _hidden_size, _hidden_size);
    }

    return SaberSuccess;
}

template<>
SaberStatus SaberAttensionLstm<NV, AK_FLOAT>:: init(const std::vector<OpTensor*>& inputs, \
                     std::vector<OpTensor*>& outputs, \
                     AttensionLstmParam <NV>& attension_lstm_param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    auto cuda_stream = ctx.get_compute_stream();

    CUBLAS_CHECK(cublasCreate(&_handle));
    CUBLAS_CHECK(cublasSetStream(_handle, cuda_stream));

    auto lstm_param = attension_lstm_param.lstm_param;
    _hidden_size = lstm_param.bias()->valid_size() / 4 / lstm_param.num_layers;
    int weights_h2h_size = _hidden_size * _hidden_size * 4 * (2 * lstm_param.num_layers - 1);
    int weights_i2h_size = lstm_param.weight()->valid_size() - weights_h2h_size;
    _word_size = weights_i2h_size / (4 * _hidden_size);
    auto fc_vec = attension_lstm_param.attension_param.fc_vec;
    _attn_outs.resize(fc_vec.size());
    _max_seq_len = 100;

    for (int i = 0; i < fc_vec.size(); i++) {
        Shape shape ({inputs[0]->num(), fc_vec[i].num_output, 1, 1});
        _attn_outs[i] = new OpTensor(shape);
    }

    return create(inputs, outputs, attension_lstm_param, ctx);
}

template<>
SaberStatus SaberAttensionLstm<NV, AK_FLOAT>::dispatch(\
        const std::vector<OpTensor*>& inputs,
        std::vector<OpTensor*>& outputs,
        AttensionLstmParam <NV>& param) {
    cudaStream_t stream = this->_ctx->get_compute_stream();
    auto attn_param = param.attension_param;
    auto lstm_param = param.lstm_param;
    int hidden_size = lstm_param.with_peephole ? lstm_param.bias()->valid_size() / 7 :
                      lstm_param.bias()->valid_size() / 4;


    OpTensor* input = inputs[0];
    _attn_outs.resize(attn_param.fc_vec.size());
    auto seq_offset = inputs[0]->get_seq_offset()[0];
    int seq_num = seq_offset.size() - 1;
    int word_num = inputs[0]->num();
    Shape softmax_out_shape ({word_num, 1, 1, 1});
    Shape dev_seq_id_shape ({seq_num, 1, 1, 1});
    _softmax_out.reshape(softmax_out_shape);
    _dev_seq_id_map.reshape(dev_seq_id_shape);
    std::vector<int> id_map;
    int seq_id = 0;
    int max_len = 0;
    for (int i = 0; i < seq_num; i++) {
        for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
            id_map.push_back(i);
        }
    }

    for (int i = 0; i < seq_num; i++) {
        int cur_len = seq_offset[i + 1] - seq_offset[i];
        max_len = max_len < cur_len ? cur_len : max_len;
    }

    cudaMemcpyAsync(_dev_seq_id_map.mutable_data(), &id_map[0], sizeof(int) * seq_num,
                    cudaMemcpyHostToDevice, stream);

    Shape offset_shape ( {seq_num + 1, 1, 1, 1});
    _dev_offset.reshape(offset_shape);
    cudaMemcpyAsync(_dev_offset.mutable_data(), &seq_offset[0],
                    sizeof(int) * seq_offset.size(),
                    cudaMemcpyHostToDevice, stream);

    /*for first fc*/


    int M_0 = input->num();
    int N_0 = attn_param.fc_vec[0].num_output;
    int K_0 = input->valid_size() / input->num();
    Shape first_fc_out_0_shape ({M_0, N_0, 1, 1});
    _first_fc_out_0.reshape(first_fc_out_0_shape);
    auto data_in = static_cast<const OpDataType*>(input->data());
    auto data_out = static_cast<OpDataType*>(_first_fc_out_0.mutable_data());
    auto fc_vec = attn_param.fc_vec;

    //auto first_fc_0_kernel = saber_find_fast_sass_gemm(false, !fc_vec[0].is_transpose_weights, M_0, N_0, K_0);
    auto first_fc_0_kernel = saber_find_fast_sass_gemm(false, false, M_0, N_0, K_0);
    first_fc_0_kernel(M_0, N_0, K_0, 1.0f, data_in, 0.f, static_cast<const OpDataType*>(fc_vec[0].weights->data()), data_out, stream);
    Shape cell_shape ({seq_num, hidden_size, 1, 1});
    _cell_out.reshape(cell_shape);
    cudaMemsetAsync(_cell_out.mutable_data(), 0, sizeof(float) * _cell_out.valid_size(), stream);
    Shape lstm_mid_shape( {seq_num, 4 * hidden_size, 1, 1});
    _hidden_out.reshape(lstm_mid_shape);
    Shape lstm_shape ({max_len * seq_num, hidden_size, 1, 1});

    _lstm_out.reshape(lstm_shape);



    /*for other fc*/
    for (int word_id = 0; word_id < max_len; word_id++) {
        _attn_outs[0]->reshape(first_fc_out_0_shape);

        if (word_id > 0) {
            Shape h_shape ({seq_num,  N_0, 1, 1});
            _first_fc_out_1.reshape(h_shape);

            auto kernel_1 = saber_find_fast_sass_gemm(false, false, seq_num, N_0, hidden_size);
            kernel_1(seq_num, N_0, hidden_size, 1.0f,
                     static_cast<const OpDataType*>(_cell_out.data()), 0.f,
                     static_cast<const OpDataType*>(fc_vec[0].weights->data()) + K_0 * N_0,  static_cast<OpDataType*>(_first_fc_out_1.mutable_data()), stream);

            sequence_bias_relu <<< CUDA_GET_BLOCKS(_attn_outs[0]->valid_size()), CUDA_NUM_THREADS, 0,
                               stream >>> (static_cast<const OpDataType*>(_first_fc_out_0.data()), static_cast<const OpDataType*>(_first_fc_out_1.data()), static_cast<const OpDataType*>(fc_vec[0].bias->data()),
                                       static_cast<const int*>(_dev_seq_id_map.data()), M_0, N_0, static_cast<OpDataType*>(_attn_outs[0]->mutable_data()));

        } else {
            cudaMemcpyAsync((void*)_attn_outs[0]->mutable_data(), (void*) _first_fc_out_0.data(),
                            sizeof(float) * _attn_outs[0]->valid_size(),
                            cudaMemcpyDeviceToDevice, stream);
            bias_relu <<< CUDA_GET_BLOCKS(_attn_outs[0]->valid_size()), CUDA_NUM_THREADS, 0,
                      stream >>> (data_out, static_cast<const OpDataType*>(fc_vec[0].bias->data()), _attn_outs[0]->valid_size(), N_0,
                              static_cast<OpDataType*>(_attn_outs[0]->mutable_data()));
        }

        for (int i = 1; i < attn_param.fc_vec.size(); i++) {
            int M = input->num();
            int N = attn_param.fc_vec[i].num_output;
            int K = attn_param.fc_vec[i - 1].num_output;
            Shape attn_out_shape ( {M, N, 1, 1});
            _attn_outs[i]->reshape(attn_out_shape);
            auto fc_in_data = static_cast<const OpDataType*>(_attn_outs[i - 1]->data());
            auto fc_out_data = static_cast<OpDataType*>(_attn_outs[i]->mutable_data());
            auto kernel = saber_find_fast_sass_gemm(false, false, M, N, K);
            kernel(M, N, K, 1.0f, fc_in_data, 0.0f, static_cast<const OpDataType*>(fc_vec[i].weights->data()), fc_out_data, stream);
            bias_relu <<< CUDA_GET_BLOCKS(_attn_outs[i]->valid_size()), CUDA_NUM_THREADS, 0,
                      stream >>> (fc_out_data, static_cast<const OpDataType*>(fc_vec[i].bias->data()), _attn_outs[i]->valid_size(), N, fc_out_data);
        }

        int fc_num = attn_param.fc_vec.size();
        int dim = inputs[0]->valid_size() / inputs[0]->num();
        Shape pool_shape( {seq_num, dim, 1, 1});
        _pool_out.reshape(pool_shape);

        sequence_softmax <<< CUDA_GET_BLOCKS(seq_num), CUDA_NUM_THREADS, 0, stream>>>
        (static_cast<const OpDataType*>(_attn_outs[fc_num - 1]->data()), static_cast<const int*>(_dev_offset.data()), seq_num, static_cast<OpDataType*>(_softmax_out.mutable_data()));

        sequence_pool <<< CUDA_GET_BLOCKS(seq_num* dim), CUDA_NUM_THREADS, 0, stream>>>(static_cast<const OpDataType*>(input->data()),
                static_cast<const OpDataType*>(_softmax_out.data()), static_cast<const int*>(_dev_offset.data()), seq_num, inputs[0]->num(), dim, static_cast<OpDataType*>(_pool_out.mutable_data()));


        auto  x_data = static_cast<const OpDataType*>(_pool_out.data());
        auto  _wx_data = static_cast<const OpDataType*>(lstm_param.weight()->data());
        auto  _bias_data = static_cast<const OpDataType*>(lstm_param.bias()->data());
        int word_size = dim;
        auto _wh_data = static_cast<const OpDataType*>(lstm_param.weight()->data()) + 4 * hidden_size *  word_size;
        _gemm_wx(seq_num, 4 * hidden_size, word_size, 1.0, x_data, 0.0, static_cast<const OpDataType*>(lstm_param.weight()->data()),
                 static_cast<OpDataType*>(_hidden_out.mutable_data()), stream);

        if (word_id > 0) {
            _gemm_wh(seq_num, 4 * hidden_size, hidden_size, 1.0,
                     static_cast<const OpDataType*>(_lstm_out.data()) + (word_id - 1) * seq_num * hidden_size, 1.0, _wh_data, static_cast<OpDataType*>(_hidden_out.mutable_data()),
                     stream);
        }

        auto cell_data = static_cast<OpDataType*>(_cell_out.mutable_data());

        lstm_bias_and_act <<< CUDA_GET_BLOCKS(seq_num* hidden_size), CUDA_NUM_THREADS, 0, stream>>>
        (static_cast<OpDataType*>(_hidden_out.data()), _bias_data, static_cast<OpDataType*>(_lstm_out.mutable_data()) + word_id * seq_num * hidden_size,
         cell_data, seq_num, hidden_size);

    }

    lstm_result_to_sequence <<< CUDA_GET_BLOCKS(word_num* hidden_size), CUDA_NUM_THREADS, 0, stream>>>
    (static_cast<OpDataType*>(_lstm_out.mutable_data()), static_cast<const int*>(_dev_seq_id_map.data()),
            static_cast<const int*>(_dev_offset.data()), seq_num, word_num, hidden_size, static_cast<OpDataType*>(outputs[0]->mutable_data()));
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberAttensionLstm, AttensionLstmParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAttensionLstm, AttensionLstmParam, NV, AK_INT8);

}
}
