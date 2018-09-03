#include "saber/funcs/impl/cuda/saber_crf_decoding.h"
#include "cuda_fp16.h"


namespace anakin{

namespace saber{

template<typename Dtype, unsigned int blockSize>
__global__ void decoding_bloclk_kernel(Dtype* decode_path, const Dtype* emission_ptr, const Dtype* trans_ptr, \
                    Dtype* alpha_ptr, int* track_ptr, int seq_len, int tag_num, const int base_idx){
    int idx = threadIdx.x;
    if (blockIdx.x == 0 && idx < tag_num){
        alpha_ptr[idx] = trans_ptr[idx] + emission_ptr[idx];
    }
    __syncthreads();
    CUDA_KERNEL_LOOP(idx, (seq_len - 1) * tag_num){
        const Dtype* x = emission_ptr + blockIdx.x * tag_num;
        const Dtype* w = trans_ptr + blockIdx.x * tag_num;
        Dtype* alpha = alpha_ptr + blockIdx.x * tag_num;
        int* track = track_ptr + blockIdx.x * tag_num;
        Dtype max_score = -1e32;//-std::numeric_limits<Dtype>::max();
        int max_index = 0;
        for (int i = 0; i < tag_num; i++){
            Dtype score = alpha[i] + w[(i + base_idx) * tag_num + idx];
            if (score > max_score) {
                max_score = score;
                max_index = i;
            }
        }
        alpha[tag_num + idx] = max_score + x[tag_num + idx];
        track[tag_num + idx] = max_index;
    }
    //__syncthreads();
}

template<typename Dtype, unsigned int blockSize>
__global__ void decoding_final_kernel(Dtype* decode_path, const Dtype* trans_ptr, Dtype* alpha_ptr, int* track_ptr, int seq_len, int tag_num){
    Dtype max_score = -1e32;
    int max_i = 0;
        for (size_t i = 0; i < tag_num; i++) {
            Dtype score = alpha_ptr[(seq_len - 1) * tag_num + i] + trans_ptr[tag_num + i];
            if (score > max_score) {
                max_score = score;
                max_i = i;
            }
        }
       decode_path[seq_len - 1] = max_i;
        for (int k = seq_len - 1; k >= 1; k--) {
            max_i = track_ptr[k * tag_num + max_i];
            decode_path[k - 1] = max_i;
        }
}

template<typename Dtype, unsigned int blockSize>
__global__ void decoding_kernel(Dtype* decode_path, const Dtype* emission_ptr, const Dtype* trans_ptr, \
                    Dtype* alpha_ptr, int* track_ptr, int seq_len, int tag_num, const int base_idx){
    int idx = threadIdx.x;
    const Dtype* x = emission_ptr;
    const Dtype* w = trans_ptr;
    Dtype* alpha_value = alpha_ptr;

    for (int i = 0; i < tag_num; ++i) alpha_value[i] = w[i] + x[i];

    for (int k = 1; k < seq_len; ++k) {
        for (int i = 0; i < tag_num; ++i) {
            Dtype max_score = -1e32;//-std::numeric_limits<Dtype>::max();
            int max_j = 0;
            for (int j = 0; j < tag_num; ++j) {
                Dtype score = alpha_value[(k - 1) * tag_num + j] +
                    w[(j + base_idx) * tag_num + i];
                if (score > max_score) {
                    max_score = score;
                    max_j = j;
                }
            }
            alpha_value[k * tag_num + i] = max_score + x[k * tag_num + i];
            track_ptr[k * tag_num + i] = max_j;
        }
    }
    Dtype max_score = -1e32;
    int max_i = 0;
    for (size_t i = 0; i < tag_num; i++) {
        Dtype score = alpha_ptr[(seq_len - 1) * tag_num + i] + trans_ptr[tag_num + i];
        if (score > max_score) {
            max_score = score;
            max_i = i;
        }
    }
    decode_path[seq_len - 1] = max_i;
    for (int k = seq_len - 1; k >= 1; k--) {
        max_i = track_ptr[k * tag_num + max_i];
        decode_path[k - 1] = max_i;
    }
}
template <>
SaberStatus SaberCrfDecoding<NV, AK_FLOAT>::dispatch( \
                        const std::vector<Tensor<NV> *>& inputs,
                        std::vector<Tensor<NV> *>& outputs,
                        CrfDecodingParam<NV>& param){
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    const OpDataType* emission_ptr = (const OpDataType*)inputs[0]->data();
    const OpDataType* trans_ptr = (const OpDataType*)param.mutable_transition_weight()->data();
    OpDataType* decode_path = (OpDataType*)outputs[0]->mutable_data();

    int tag_num = inputs[0]->channel();
    int slice_size = tag_num * inputs[0]->height() * inputs[0]->width();
    std::vector<std::vector<int>> seq_offset = inputs[0]->get_seq_offset();
    int seq_num = seq_offset[0].size() - 1;
    const int base_idx = 2;
    for (int i = 0; i < seq_num; i++){
        int seq_len = seq_offset[0][i+1] - seq_offset[0][i];
        if (seq_len < 1) continue;
        #if 0
        //const int threads = CUDA_NUM_THREADS >= tag_num ? tag_num : CUDA_NUM_THREADS;
        decoding_block_kernel<OpDataType, CUDA_NUM_THREADS><<<seq_len, tag_num, 0, \
            cuda_stream>>>(decode_path, emission_ptr, trans_ptr, (OpDataType*)_alpha.mutable_data(), \
            (int*)_track.mutable_data(), seq_len, tag_num, base_idx);
        decoding_final_kernel<OpDataType, 1><<<1, 1, 0, cuda_stream>>>(decode_path, trans_ptr, (OpDataType*)_alpha.mutable_data(), \
            (int*)_track.mutable_data(), seq_len, tag_num);
        #endif
        decoding_kernel<OpDataType, 1><<<1, 1, 0, cuda_stream>>>(decode_path, \
            emission_ptr, trans_ptr, (OpDataType*)_alpha.mutable_data(), \
            (int*)_track.mutable_data(), seq_len, tag_num, base_idx);

        emission_ptr += slice_size * seq_len;
        decode_path += seq_len;
    }
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberCrfDecoding, CrfDecodingParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberCrfDecoding, CrfDecodingParam, NV, AK_INT16);
} //namespace anakin

} //namespace anakin
