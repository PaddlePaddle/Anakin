#include "saber/funcs/impl/cuda/saber_gru.h"
#include "saber/core/tensor_op.h"
#include "cuda_fp16.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
namespace anakin {

namespace saber {

////TODO:can try record vector in shared
template <typename Dtype>
__global__ void trans_map2in(Dtype* output, const Dtype* input, const int* map, int count,
                             int lastdim) {
    CUDA_KERNEL_LE(tid, count) {
        int seq = tid / lastdim;
        output[tid] = input[map[seq] * lastdim + tid % lastdim];
    }
}

template <typename Dtype>
__global__ void trans_map2out(Dtype* output, const Dtype* input, const int* map, int count,
                              int lastdim) {
    CUDA_KERNEL_LE(tid, count) {
        int seq = tid / lastdim;
        output[map[seq]*lastdim + tid % lastdim] = input[tid];
    }
}

template <>
void SaberGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::seq2hw(\
        std::vector<DataTensor_out*> outputs, std::vector<DataTensor_in*> inputs,
        GruParam<OpTensor>& param, int hidden_size,
        void* real_temp_out
                                                                         ) {
    DataTensor_in* din = inputs[0];
    DataTensor_out* dout = outputs[0];
    int wordsize = din->channel();
    std::vector<int> offset_vec = din->get_seq_offset();
    CHECK_GE(offset_vec.size(), 2) << "offset must >=2" ;
    int batch_size = offset_vec.size() - 1;

    int max_len = 0;
    std::vector<int> length_vec;

    if ((void*)(outputs[0]->data()) == real_temp_out) {
        DLOG(INFO) << "not use inner space";
        return;
    }

    const OutDataType* origin = _temp_tensor_out.data();
    OutDataType* target = dout->mutable_data();

    //source is sequence id in seq target is hw id in seq,map is source to target ptr offset
    int seq_sum = offset_vec[batch_size];
    CUDA_CHECK(cudaMemcpyAsync(_temp_map_dev.mutable_data(), _temp_map_host.data(), sizeof(int)*seq_sum,
                               cudaMemcpyHostToDevice, _ctx.get_compute_stream()));
    int count=seq_sum * hidden_size;
    int block_dim=count;
    int grid_dim=1;
    if(count>1024){
        block_dim=256;
        grid_dim=(count+block_dim-1)/block_dim;
    }
    trans_map2in <<< grid_dim, block_dim, 0, _ctx.get_compute_stream()>>>(target, origin, _temp_map_dev.data(),
            count, hidden_size);

//    trans_map2in_old <<< 4, 128, 0, _ctx.get_compute_stream()>>>(target, origin, _temp_map_dev.data(),
//            count, hidden_size);

}

//TODO:gem by self, flatten by time, padding by nothing (zhangs)
template <>
const float* SaberGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::hw2seq(\
        std::vector<DataTensor_in*> inputs, GruParam<OpTensor>& param, \
        int word_size, int hidden_size, int& sequence_len) {
    DataTensor_in* din = inputs[0];

    std::vector<int> offset_vec = din->get_seq_offset();
    CHECK_GE(offset_vec.size(), 2) << "offset must >=2" ;
    int batch_size = offset_vec.size() - 1;
    int seq_sum = offset_vec[offset_vec.size() - 1];
    int wordsize = din->channel();
    int max_len = 0;
    std::vector<int> length_vec(batch_size);

    for (int i = 0; i < offset_vec.size() - 1; ++i) {
        int len = offset_vec[i + 1] - offset_vec[i];
        max_len = max_len > len ? max_len : len;
        length_vec[i] = len;
    }

    Shape seq_shape(1, max_len, batch_size, word_size);
    _temp_tensor_in.try_expand_size(seq_shape);

    Shape seq_out_shape(1, max_len, batch_size, hidden_size);
    _temp_tensor_out.try_expand_size(seq_out_shape);

    sequence_len = max_len;

    if (batch_size == 1 || max_len == 1) {
        return din->mutable_data();
    }

    InDataType* target = _temp_tensor_in.mutable_data();
    const InDataType* origin = din->data();

    _temp_map_host.try_expand_size(seq_sum);
    _temp_map_dev.try_expand_size(seq_sum);
    int* map = _temp_map_host.mutable_data();

    if (param._is_reverse) {
        for (int batchid = 0; batchid < batch_size; ++batchid) {
            int batch_offset = max_len - length_vec[batchid];

            for (int seqid = 0; seqid < length_vec[batchid]; ++seqid) {
                int source = (offset_vec[batchid] + seqid);
                int target = ((seqid + batch_offset) * batch_size + batchid);
                map[source] = target;
            }
        }
    } else {
        for (int batchid = 0; batchid < batch_size; ++batchid) {
            for (int seqid = 0; seqid < length_vec[batchid]; ++seqid) {
                int source = (offset_vec[batchid] + seqid);
                int target = (seqid * batch_size + batchid);
                map[source] = target;
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(_temp_map_dev.mutable_data(), _temp_map_host.data(), sizeof(int)*seq_sum,
                               cudaMemcpyHostToDevice, _ctx.get_compute_stream()));
    int count=seq_sum * wordsize;
    int block_dim=count;
    int grid_dim=1;
    if(count>1024){
        block_dim=256;
        grid_dim=(count+block_dim-1)/block_dim;
    }
    trans_map2out <<< grid_dim, block_dim, 0, _ctx.get_compute_stream()>>>(target, origin, _temp_map_dev.data(),
            count, wordsize);

//    trans_map2out_old <<< 4, 128, 0, _ctx.get_compute_stream()>>>(target, origin, _temp_map_dev.data(),
//            count, wordsize);


    return _temp_tensor_in.data();
}

#define SIGMOID_THRESHOLD_MIN_PADDLE -40.0
#define SIGMOID_THRESHOLD_MAX_PADDLE 13.0
#define EXP_MAX_INPUT_PADDLE 40.0

template <typename T>
inline  static __device__ T identity(const T a) {
    return a;
}

template <typename T>
inline static __device__ T relu(const T a) {
    return a > static_cast<T>(0.0) ? a : static_cast<T>(0.0);
}


template <typename T>
inline static __device__ T sigmoid_paddle(const T a) {
    const T min = SIGMOID_THRESHOLD_MIN_PADDLE;
    const T max = SIGMOID_THRESHOLD_MAX_PADDLE;
    T tmp = (a < min) ? min : ((a > max) ? max : a);
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-tmp));
}

template <typename T>
inline static __device__ T tanh_paddle(const T a) {
    T tmp = -2.0 * a;
    tmp = (tmp > EXP_MAX_INPUT_PADDLE) ? EXP_MAX_INPUT_PADDLE : tmp;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

static void anakin_NV_gemm(cublasHandle_t handle, const bool TransA,
                           const bool TransB, const int M, const int N, const int K,
                           const float alpha, const float* A, const float* B, const float beta,
                           float* C) {
    // Note that cublas follows fortran order.
    int lda = (!TransA/* == CblasNoTrans*/) ? K : M;
    int ldb = (!TransB/* == CblasNoTrans*/) ? N : K;
    cublasOperation_t cuTransA =
        (!TransA/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (!TransB/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
                             N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

/**
 * gridDim=batchsize
 * @tparam Dtype
 * @param w_x_r
 * @param w_h_r
 * @param br
 * @param hidden_size
 * @param output_r
 * @param w_x_z
 * @param w_h_z
 * @param bz
 * @param output_z
 */
template <typename Dtype>
__global__ void cal_reset_update(Dtype* w_x_r, Dtype* w_h_r, const Dtype* b_r,
                                 const int hidden_size, Dtype* output_r,
                                 Dtype* w_x_z, Dtype* w_h_z, const Dtype* b_z, Dtype* output_z) {
    int w_base_index = blockIdx.x * hidden_size * 3;
    int h_base_index = blockIdx.x * hidden_size;
    Dtype* in_w_x_r = w_x_r + w_base_index;
    Dtype* in_w_h_r = w_h_r + w_base_index;
    Dtype* in_w_x_z = w_x_z + w_base_index;
    Dtype* in_w_h_z = w_h_z + w_base_index;
    Dtype* out_output_r = output_r + h_base_index;
    Dtype* out_output_z = output_z + h_base_index;

    for (int index = threadIdx.x; index < hidden_size; index += blockDim.x) {
        Dtype before_act_r = in_w_x_r[index] + in_w_h_r[index] + b_r[index];
        out_output_r[index] = Dtype(Dtype(1) / (Dtype(1) + expf(-before_act_r)));
        Dtype before_act_z = in_w_x_z[index] + in_w_h_z[index] + b_z[index];
        out_output_z[index] = Dtype(Dtype(1) / (Dtype(1) + expf(-before_act_z)));

    }
}

template <typename Dtype>
__global__ void cal_final(Dtype* w_x_o, Dtype* w_h_o, Dtype* reset, const Dtype* b_o,
                          const int hidden_size, Dtype* update, Dtype* output, Dtype* hidden_pre) {
    int w_base_index = blockIdx.x * hidden_size * 3;
    int h_base_index = blockIdx.x * hidden_size;

    Dtype* in_w_x_o = w_x_o + w_base_index;
    Dtype* in_w_h_o = w_h_o + w_base_index;
    Dtype* in_hidden_pre = hidden_pre + h_base_index;
    Dtype* in_update = update + h_base_index;
    Dtype* in_reset = reset + h_base_index;
    Dtype* out_output = output + h_base_index;

    for (int index = threadIdx.x; index < hidden_size; index += blockDim.x) {
        Dtype before_act_h = in_w_x_o[index] + in_w_h_o[index] * in_reset[index]
                             + b_o[index];
        Dtype acted = tanhf(before_act_h);
        Dtype update_t = in_update[index];
        out_output[index] = (1 - update_t) * acted + update_t* in_hidden_pre[index];
    }
}

template <typename Dtype>
__global__ void cal_one_kernel_paddlesigmoid_tanh_cudnn_formula(Dtype* w_x_r, Dtype* w_x_z,
        Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, Dtype* w_h_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre) {
    int w_base_index = blockIdx.x * hidden_size * 3;
    int h_base_index = blockIdx.x * hidden_size;
    Dtype* in_w_x_r = w_x_r + w_base_index;
    Dtype* in_w_h_r = w_h_r + w_base_index;
    Dtype* in_w_x_z = w_x_z + w_base_index;
    Dtype* in_w_h_z = w_h_z + w_base_index;
    Dtype* in_w_x_o = w_x_o + w_base_index;
    Dtype* in_w_h_o = w_h_o + w_base_index;
    const Dtype* in_hidden_pre = hidden_pre + h_base_index;
    Dtype* out_output = output + h_base_index;

    for (int index = threadIdx.x; index < hidden_size; index += blockDim.x) {
        const Dtype min = SIGMOID_THRESHOLD_MIN_PADDLE;
        const Dtype max = SIGMOID_THRESHOLD_MAX_PADDLE;

        Dtype before_act_r = in_w_x_r[index] + in_w_h_r[index] + b_r[index];
        before_act_r = (before_act_r < min) ? min : ((before_act_r > max) ? max : before_act_r);
        Dtype act_r = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-before_act_r));

        Dtype before_act_z = in_w_x_z[index] + in_w_h_z[index] + b_z[index];
        before_act_z = (before_act_z < min) ? min : ((before_act_z > max) ? max : before_act_z);
        Dtype act_z = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-before_act_z));

        Dtype before_act_h = in_w_x_o[index] + in_w_h_o[index] * act_r
                             + b_o[index];
        before_act_h = (before_act_h > EXP_MAX_INPUT_PADDLE) ? EXP_MAX_INPUT_PADDLE : before_act_h;
        Dtype acted = tanhf(before_act_h);
        out_output[index] = (1 - act_z) * acted + act_z * in_hidden_pre[index];
    }
}

template <typename Dtype>
__global__ void cal_one_kernel_sigmoid_tanh_modi_cudnn_formula(Dtype* w_x_r, Dtype* w_x_z,
        Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, Dtype* w_h_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre) {

    int w_base_index = blockIdx.x * hidden_size * 3 + threadIdx.x;
    int h_base_index = blockIdx.x * hidden_size + threadIdx.x;

    for (int index = threadIdx.x; index < hidden_size;
            index += blockDim.x, w_base_index += blockDim.x, h_base_index += blockDim.x) {
        Dtype before_act_r = w_x_r[w_base_index] + w_h_r[w_base_index] + b_r[index];
        Dtype act_r = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_r));
        Dtype before_act_z = w_x_z[w_base_index] + w_h_z[w_base_index] + b_z[index];
        Dtype act_z = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_z));
        Dtype before_act_h = w_x_o[w_base_index] + w_h_o[w_base_index] * act_r
                             + b_o[index];
        Dtype acted = tanh(before_act_h);
        output[h_base_index] = (static_cast<Dtype>(1.0) - act_z) * acted + act_z * hidden_pre[h_base_index];
    }
}

template <typename Dtype>
__global__ void cal_one_kernel_paddlesigmoid_relu_paddle_formula(Dtype* w_x_r, Dtype* w_x_z,
        Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, const Dtype* w_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre) {
    int index = threadIdx.x;

    if (index > hidden_size) {
        return;
    }

    int w_base_index = blockIdx.x * hidden_size * 3 + index;
    int u_base_index = blockIdx.x * hidden_size * 2 + index;
    int h_base_index = blockIdx.x * hidden_size + index;
    extern __shared__ Dtype shared_hidden_pre[];
    Dtype hidden_pre_value = hidden_pre[h_base_index];
    Dtype before_act_r = w_x_r[w_base_index] + w_h_r[u_base_index] + b_r[index];
    Dtype act_r = sigmoid_paddle(before_act_r);
    shared_hidden_pre[index] = hidden_pre_value * act_r;
    Dtype before_act_z = w_x_z[w_base_index] + w_h_z[u_base_index] + b_z[index];
    Dtype act_z = sigmoid_paddle(before_act_z);
    Dtype w_h_o = static_cast<Dtype>(0.0);
    int k_index = index;
    __syncthreads();

    for (int w_index = 0; w_index < hidden_size; ++w_index) {
        w_h_o += shared_hidden_pre[w_index] * w_o[k_index];
        k_index += hidden_size;
    }

    Dtype before_act_h = w_x_o[w_base_index] + w_h_o
                         + b_o[index];
    Dtype acted = relu(before_act_h);
    output[h_base_index] = (static_cast<Dtype>(1.0) - act_z) * hidden_pre_value + act_z * acted;
}

template <typename Dtype>
__global__ void cal_one_kernel_sigmoid_tanh_paddle_formula(Dtype* w_x_r, Dtype* w_x_z, Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, const Dtype* w_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre) {
    int index = threadIdx.x;

    if (index > hidden_size) {
        return;
    }

    int w_base_index = blockIdx.x * hidden_size * 3 + index;
    int u_base_index = blockIdx.x * hidden_size * 2 + index;
    int h_base_index = blockIdx.x * hidden_size + index;
    extern __shared__ Dtype shared_hidden_pre[];
    Dtype hidden_pre_value = hidden_pre[h_base_index];
    Dtype before_act_r = w_x_r[w_base_index] + w_h_r[u_base_index] + b_r[index];

    Dtype act_r = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_r));
//    printf("%d %f=[%f , %f ,%f]\n",index,act_r,w_x_r[w_base_index],w_h_r[u_base_index],b_r[index]);
    shared_hidden_pre[index] = hidden_pre_value * act_r;
    Dtype before_act_z = w_x_z[w_base_index] + w_h_z[u_base_index] + b_z[index];
    Dtype act_z = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_z));
    Dtype w_h_o = static_cast<Dtype>(0.0);
    int k_index = index;
    __syncthreads();

    for (int w_index = 0; w_index < hidden_size; ++w_index) {
        w_h_o += shared_hidden_pre[w_index] * w_o[k_index];
        k_index += hidden_size;
    }

    Dtype before_act_h = w_x_o[w_base_index] + w_h_o
                         + b_o[index];
    Dtype acted = tanhf(before_act_h);
    output[h_base_index] = (static_cast<Dtype>(1.0) - act_z) * hidden_pre_value + act_z * acted;
//    printf("output %d = %f\n",index,output[h_base_index]);
}


template <typename Dtype>
__global__ void cal_one_kernel_sigmoid_tanh(Dtype* w_x_r, Dtype* w_x_z, Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, Dtype* w_h_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre) {
    int w_base_index = blockIdx.x * hidden_size * 3;
    int h_base_index = blockIdx.x * hidden_size;
    Dtype* in_w_x_r = w_x_r + w_base_index;
    Dtype* in_w_h_r = w_h_r + w_base_index;
    Dtype* in_w_x_z = w_x_z + w_base_index;
    Dtype* in_w_h_z = w_h_z + w_base_index;
    Dtype* in_w_x_o = w_x_o + w_base_index;
    Dtype* in_w_h_o = w_h_o + w_base_index;
    const Dtype* in_hidden_pre = hidden_pre + h_base_index;
    Dtype* out_output = output + h_base_index;

    for (int index = threadIdx.x; index < hidden_size; index += blockDim.x) {
        Dtype before_act_r = in_w_x_r[index] + in_w_h_r[index] + b_r[index];
        Dtype act_r = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_r));
        Dtype before_act_z = in_w_x_z[index] + in_w_h_z[index] + b_z[index];
        Dtype act_z = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_z));
        Dtype before_act_h = in_w_x_o[index] + in_w_h_o[index] * act_r
                             + b_o[index];
        Dtype acted = tanhf(before_act_h);
        out_output[index] = (static_cast<Dtype>(1.0) - act_z) * acted + act_z * in_hidden_pre[index];
    }
}

template <typename Dtype>
__global__ void cal_one_kernel_sigmoid_tanh_index_modi(Dtype* w_x_r, Dtype* w_x_z, Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, Dtype* w_h_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre,
        int seq_batch_hidden, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= seq_batch_hidden) {
        return;
    }

    int batch_id = tid / hidden_size % batch_size;
    int index = tid % hidden_size;
    int w_base_index = batch_id * hidden_size * 3;
    int h_base_index = batch_id * hidden_size;
    int index_w = index + w_base_index;
    int index_h = index + h_base_index;

    {
        Dtype before_act_r = w_x_r[index_w] + w_h_r[index_w] + b_r[index];
        Dtype act_r = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_r));
        Dtype before_act_z = w_x_z[index_w] + w_h_z[index_w] + b_z[index];
        Dtype act_z = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_z));
        Dtype before_act_h = w_x_o[index_w] + w_h_o[index_w] * act_r
                             + b_o[index];
        Dtype acted = tanhf(before_act_h);
        output[index_h] = (static_cast<Dtype>(1.0) - act_z) * acted + act_z * hidden_pre[index_h];
    }
}

template <typename Dtype>
__global__ void cal_one_kernel_sigmoid_tanh_index(Dtype* w_x_r, Dtype* w_x_z, Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, Dtype* w_h_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre,
        int seq_batch_hidden, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= seq_batch_hidden) {
        return;
    }

    int batch_id = tid / hidden_size % batch_size;
    int index = tid % hidden_size;
    int w_base_index = batch_id * hidden_size * 3;
    int h_base_index = batch_id * hidden_size;
    Dtype* in_w_x_r = w_x_r + w_base_index;
    Dtype* in_w_h_r = w_h_r + w_base_index;
    Dtype* in_w_x_z = w_x_z + w_base_index;
    Dtype* in_w_h_z = w_h_z + w_base_index;
    Dtype* in_w_x_o = w_x_o + w_base_index;
    Dtype* in_w_h_o = w_h_o + w_base_index;
    const Dtype* in_hidden_pre = hidden_pre + h_base_index;
    Dtype* out_output = output + h_base_index;
    {
        Dtype before_act_r = in_w_x_r[index] + in_w_h_r[index] + b_r[index];
        Dtype act_r = Dtype(Dtype(1) / (Dtype(1) + expf(-before_act_r)));
        Dtype before_act_z = in_w_x_z[index] + in_w_h_z[index] + b_z[index];
        Dtype act_z = Dtype(Dtype(1) / (Dtype(1) + expf(-before_act_z)));
        Dtype before_act_h = in_w_x_o[index] + in_w_h_o[index] * act_r
                             + b_o[index];
        Dtype acted = tanhf(before_act_h);
        out_output[index] = (1 - act_z) * acted + act_z * in_hidden_pre[index];
    }
}
template <typename Dtype>
__global__ void cal_one_kernel_paddlesigmoid_relu_cudnn_formula(Dtype* w_x_r, Dtype* w_x_z,
        Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, Dtype* w_h_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre) {
    int w_base_index = blockIdx.x * hidden_size * 3;
    int h_base_index = blockIdx.x * hidden_size;
    Dtype* in_w_x_r = w_x_r + w_base_index;
    Dtype* in_w_h_r = w_h_r + w_base_index;
    Dtype* in_w_x_z = w_x_z + w_base_index;
    Dtype* in_w_h_z = w_h_z + w_base_index;
    Dtype* in_w_x_o = w_x_o + w_base_index;
    Dtype* in_w_h_o = w_h_o + w_base_index;
    const Dtype* in_hidden_pre = hidden_pre + h_base_index;
    Dtype* out_output = output + h_base_index;

    for (int index = threadIdx.x; index < hidden_size; index += blockDim.x) {
        const Dtype min = SIGMOID_THRESHOLD_MIN_PADDLE;
        const Dtype max = SIGMOID_THRESHOLD_MAX_PADDLE;

        Dtype before_act_r = in_w_x_r[index] + in_w_h_r[index] + b_r[index];
        before_act_r = (before_act_r < min) ? min : ((before_act_r > max) ? max : before_act_r);
        Dtype act_r = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-before_act_r));

        Dtype before_act_z = in_w_x_z[index] + in_w_h_z[index] + b_z[index];
        before_act_z = (before_act_z < min) ? min : ((before_act_z > max) ? max : before_act_z);
        Dtype act_z = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-before_act_z));

        Dtype before_act_h = in_w_x_o[index] + in_w_h_o[index] * act_r
                             + b_o[index];

        Dtype acted = before_act_h > static_cast<Dtype>(0.0) ? before_act_h : static_cast<Dtype>(0.0);
        out_output[index] = (1 - act_z) * acted + act_z * in_hidden_pre[index];

    }
}

template <typename Dtype>
__global__ void cal_one_kernel_sigmoid_tanh_modi(Dtype* w_x_r, Dtype* w_x_z, Dtype* w_x_o,
        Dtype* w_h_r, Dtype* w_h_z, Dtype* w_h_o,
        const Dtype* b_r, const Dtype* b_z, const Dtype* b_o,
        int hidden_size, Dtype* output, const Dtype* hidden_pre) {

    int w_base_index = blockIdx.x * hidden_size * 3 + threadIdx.x;
    int h_base_index = blockIdx.x * hidden_size + threadIdx.x;

    for (int index = threadIdx.x; index < hidden_size;
            index += blockDim.x, w_base_index += blockDim.x, h_base_index += blockDim.x) {
        Dtype before_act_r = w_x_r[w_base_index] + w_h_r[w_base_index] + b_r[index];

        Dtype act_r = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_r));
        Dtype before_act_z = w_x_z[w_base_index] + w_h_z[w_base_index] + b_z[index];
        Dtype act_z = static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-before_act_z));
        Dtype before_act_h = w_x_o[w_base_index] + w_h_o[w_base_index] * act_r
                             + b_o[index];
        Dtype acted = tanhf(before_act_h);
        output[h_base_index] = (static_cast<Dtype>(1.0) - act_z) * acted + act_z * hidden_pre[h_base_index];
    }
}

template <>
SaberStatus SaberGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::gru_cudnn(
    const std::vector<DataTensor_in*> inputs,
    std::vector<DataTensor_out*> outputs,
    GruParam<OpTensor>& param) {

    DataTensor_in* x = inputs[0];
    const InDataType* x_data = x->data();
    const InDataType* h;
    DataTensor_out* dout = outputs[0];
    OutDataType* dout_data = dout->mutable_data();

    //TODO:check shape first
    const OpTensor* b = param.bias();

    int batch_size = inputs[0]->get_seq_offset().size() - 1;; //x->get_seq_offset().size()-1;
    int sequence = x->num();
    int hidden_size = b->valid_size() / 3;
    bool isHW2Seq=inputs[0]->get_seq_offset().size()>2;
    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;

//    CHECK_EQ(w_h2h->height(), hidden_size) << "w_h2h->height()==batch_size";
//    CHECK_EQ(w_h2h->width(), hidden_size * 3) << "w_h2h->width()==hidden_size*3";
//
//    CHECK_EQ(w_i2h->height(), word_size) << "w_i2h->height()==word_size";
//    CHECK_EQ(w_i2h->width(), hidden_size * 3) << "w_i2h->width()==hidden_size*3";

    if (isHW2Seq) {
        x_data = hw2seq(inputs, param, _word_size, hidden_size, sequence);
        batch_size = inputs[0]->get_seq_offset().size() - 1;

        if (x_data != x->data()) {
            dout_data = _temp_tensor_out.mutable_data();
        }
    }

    Shape shape_wx(sequence, batch_size, 3, hidden_size);
    _temp_WX.try_expand_size(shape_wx);

    Shape shape_wh(1, batch_size, 3, hidden_size);
    _temp_WH.try_expand_size(shape_wh);

    anakin_NV_gemm(_cublas_handle, false, false, sequence * batch_size, 3 * hidden_size,
                   _word_size, 1.0, x_data, _weights_i2h.data(), 0.0, _temp_WX.mutable_data());


    const OpDataType* b_r = b->data() + r_offset * hidden_size;
    const OpDataType* b_z = b->data() + z_offset * hidden_size;
    const OpDataType* b_o = b->data() + o_offset * hidden_size;

    if (inputs.size() == 1) {
        CUDA_CHECK(cudaMemsetAsync(dout_data, 0, sizeof(InDataType) * batch_size * hidden_size,
                                   _ctx.get_compute_stream()));
        h = dout_data;
    } else {
        h = inputs[1]->data();
        CHECK_EQ(inputs[1]->valid_size(), batch_size * hidden_size) <<
                "h size should be batch_size * hidden_size";
    }

    for (int seq = 0; seq < sequence; seq++) {
        const InDataType* hidden_in;
        InDataType* hidden_out = dout_data + seq * batch_size * hidden_size;

        if (seq == 0) {
            hidden_in = h;
        } else {
            hidden_in = dout_data + (seq - 1) * batch_size * hidden_size;
        }

        anakin_NV_gemm(_cublas_handle, false, false, batch_size,
                       3 * hidden_size, hidden_size, 1.0, hidden_in,
                       _weights_h2h.data(), 0.0, _temp_WH.mutable_data());

        OpDataType* w_x_r = _temp_WX.mutable_data() + r_offset * hidden_size
                            + seq * batch_size * hidden_size * 3;
        OpDataType* w_x_z = _temp_WX.mutable_data() + z_offset * hidden_size
                            + seq * batch_size * hidden_size * 3;
        OpDataType* w_x_o = _temp_WX.mutable_data() + o_offset * hidden_size
                            + seq * batch_size * hidden_size * 3;

        OpDataType* w_h_r = _temp_WH.mutable_data() + r_offset * hidden_size;
        OpDataType* w_h_z = _temp_WH.mutable_data() + z_offset * hidden_size;
        OpDataType* w_h_o = _temp_WH.mutable_data() + o_offset * hidden_size;

        int frame_per_block = hidden_size <= 1024 ? hidden_size : 1024;

        if (param._gate_activity == Active_sigmoid
                && param._h_activity == Active_tanh) {
            cal_one_kernel_sigmoid_tanh_modi_cudnn_formula
                    << < batch_size, frame_per_block, 0, _ctx.get_compute_stream() >> >
                    (w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_h_o
                     , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);
        } else if (param._gate_activity == Active_sigmoid_fluid
                   && param._h_activity == Active_tanh) {
            cal_one_kernel_paddlesigmoid_tanh_cudnn_formula
                    << < batch_size, frame_per_block, 0, _ctx.get_compute_stream() >> >
                    (w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_h_o
                     , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);
        } else if (param._gate_activity == Active_sigmoid_fluid
                   && param._h_activity == Active_relu) {
            cal_one_kernel_paddlesigmoid_relu_cudnn_formula
                    << < batch_size, frame_per_block, 0, _ctx.get_compute_stream() >> >
                    (w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_h_o
                     , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);
        } else {
            LOG(ERROR) << "not support active  function";
        }

    }

    if (isHW2Seq) {
        seq2hw(outputs, inputs, param, hidden_size, dout_data);
        outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    }


}
template<>
SaberStatus SaberGru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        GruParam <OpTensor>& param) {
    if (param._formula == GRU_CUDNN) {
        LOG(ERROR) << "saber cudnn formula not support reverse yet";
        if (param._is_reverse) {
            LOG(ERROR) << "saber cudnn formula not support reverse yet";

        }
        return gru_cudnn(inputs, outputs, param);
    }

    //    LOG(INFO)<<"gru_paddle";
    DataTensor_in* x = inputs[0];
    const InDataType* x_data = x->data();
    const InDataType* h;
    DataTensor_out* dout = outputs[0];
    OutDataType* dout_data = dout->mutable_data();

    //TODO:check shape first
    const OpTensor* b = param.bias();

    int batch_size = x->get_seq_offset().size() - 1; //x->get_seq_offset().size()-1;
    int sequence = x->num();
    int hidden_size = b->valid_size() / 3;
    bool isHW2Seq=inputs[0]->get_seq_offset().size()>2;
    int o_offset = 0;
    int r_offset = 1;
    int z_offset = 2;

//    CHECK_EQ(w_h2h->height(), hidden_size) << "w_h2h->height()==batch_size";
//    CHECK_EQ(w_h2h->width(), hidden_size * 3) << "w_h2h->width()==hidden_size*3";
//
//    CHECK_EQ(w_i2h->height(), word_size) << "w_i2h->height()==word_size";
//    CHECK_EQ(w_i2h->width(), hidden_size * 3) << "w_i2h->width()==hidden_size*3";

    if (isHW2Seq) {
        x_data = hw2seq(inputs, param, _word_size, hidden_size, sequence);
//        batch_size = inputs[0]->get_seq_offset().size() - 1;

        if (x_data != x->data()) {
            dout_data = _temp_tensor_out.mutable_data();
        }
    }

    Shape shape_WX(sequence, batch_size, 3, hidden_size);
    _temp_WX.try_expand_size(shape_WX);

    Shape shape_WH(1, batch_size, 2, hidden_size);
    _temp_WH.try_expand_size(shape_WH);

    anakin_NV_gemm(_cublas_handle, false, false, sequence * batch_size, 3 * hidden_size,
                   _word_size, 1.0, x_data, _weights_i2h.data(), 0.0, _temp_WX.mutable_data());

    const OpDataType* b_r = b->data() + r_offset * hidden_size;
    const OpDataType* b_z = b->data() + z_offset * hidden_size;
    const OpDataType* b_o = b->data() + o_offset * hidden_size;

    if (inputs.size() == 1) {
        CUDA_CHECK(cudaMemsetAsync(dout_data, 0, sizeof(OutDataType)*batch_size * hidden_size,
                                   _ctx.get_compute_stream()));
        h = dout_data;
    } else {
        h = inputs[1]->data();
    }

    for (int seq = 0; seq < sequence; ++seq) {
        int realseq = seq;
        int last_seq = realseq - 1;

        if (param._is_reverse) {
//            DLOG(INFO)<<"reverse gru";
            realseq = sequence - 1 - seq;
            last_seq = realseq + 1;
        }

        const OutDataType* hidden_in;
        OutDataType* hidden_out = dout_data + realseq * batch_size * hidden_size;

        if (seq == 0) {
            hidden_in = h;
        } else {
            hidden_in = dout_data + last_seq * batch_size * hidden_size;
        }

        anakin_NV_gemm(_cublas_handle, false, false, batch_size,
                       2 * hidden_size, hidden_size, 1.0, hidden_in,
                       _weights_h2h.data() + hidden_size * hidden_size, 0.0, _temp_WH.mutable_data());


        OutDataType* w_x_r = _temp_WX.mutable_data() + r_offset * hidden_size
                             + realseq * batch_size * hidden_size * 3;
        OutDataType* w_x_z = _temp_WX.mutable_data() + z_offset * hidden_size
                             + realseq * batch_size * hidden_size * 3;
        OutDataType* w_x_o = _temp_WX.mutable_data() + o_offset * hidden_size
                             + realseq * batch_size * hidden_size * 3;

        OutDataType* w_h_r = _temp_WH.mutable_data() + 0 * hidden_size;
        OutDataType* w_h_z = _temp_WH.mutable_data() + 1 * hidden_size;
        const OpDataType * w_o = _weights_h2h.data();

        CHECK_LE(hidden_size, 1024) << "now not support hidden size > 1024 for paddle formula";

        int frame_per_block = hidden_size <= 1024 ? hidden_size : 1024;

        //        DLOG(INFO) << "act = " << param._gate_activity << "," << param._h_activity;

        if (param._gate_activity == Active_sigmoid
                && param._h_activity == Active_tanh) {
            cal_one_kernel_sigmoid_tanh_paddle_formula
            <<< batch_size, frame_per_block, sizeof(OutDataType)*hidden_size
            , _ctx.get_compute_stream()>>>(
                w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_o
                , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);

        }  else if (param._gate_activity == Active_sigmoid_fluid
                    && param._h_activity == Active_relu) {
            cal_one_kernel_paddlesigmoid_relu_paddle_formula
                    << < batch_size, frame_per_block, sizeof(OutDataType)*hidden_size
                    , _ctx.get_compute_stream() >> >
                    (w_x_r, w_x_z, w_x_o, w_h_r, w_h_z, w_o
                     , b_r, b_z, b_o, hidden_size, hidden_out, hidden_in);

        } else {
            LOG(ERROR) << "not support active  function";
        }

    }

    if (isHW2Seq) {
        seq2hw(outputs, inputs, param, hidden_size, dout_data);
    }
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

}


}
}

