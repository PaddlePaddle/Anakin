#include "saber/funcs/impl/cuda/saber_cos_sim.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_cos_sim_fwd(Dtype * out_data,
                                const Dtype* in_0,
                                const Dtype* in_1,
                                const int num,
                                const int len,
                                const float epsilon) {
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    extern __shared__ Dtype share_mem[];
    Dtype* aa_sum = share_mem;
    Dtype* bb_sum = share_mem + blockDim.x;
    Dtype* ab_sum = bb_sum + blockDim.x;
    aa_sum[thread_idx] = 0;
    bb_sum[thread_idx] = 0;
    ab_sum [thread_idx] = 0;
    const Dtype* in_0_tmp = in_0 + block_idx * len;
    const Dtype* in_1_tmp = in_1 + block_idx * len;
    for (int i = thread_idx; i < len; i += blockDim.x) {
        aa_sum[thread_idx] += in_0_tmp[i] * in_0_tmp[i];
        bb_sum[thread_idx] += in_1_tmp[i] * in_1_tmp[i];
        ab_sum[thread_idx] += in_0_tmp[i] * in_1_tmp[i];
    }
    __syncthreads();
    if (blockDim.x >= 512) {
        if (thread_idx < 256) {
            int index = thread_idx + 256;
            aa_sum[thread_idx] += aa_sum[index];
            bb_sum[thread_idx] += bb_sum[index];
            ab_sum[thread_idx] += ab_sum[index];
        }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (thread_idx < 128) {
            int index = thread_idx + 128;
            aa_sum[thread_idx] += aa_sum[index];
            bb_sum[thread_idx] += bb_sum[index];
            ab_sum[thread_idx] += ab_sum[index];
        }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (thread_idx < 64) {
            int index = thread_idx + 64;
            aa_sum[thread_idx] += aa_sum[index];
            bb_sum[thread_idx] += bb_sum[index];
            ab_sum[thread_idx] += ab_sum[index];
        }
        __syncthreads();
    }
    if (blockDim.x >= 64) {
        if (thread_idx < 32) {
            int index = thread_idx + 32;
            aa_sum[thread_idx] += aa_sum[index];
            bb_sum[thread_idx] += bb_sum[index];
            ab_sum[thread_idx] += ab_sum[index];
        }
        __syncthreads();
    }
    if (blockDim.x >= 32) {
        volatile Dtype *vaa_sum = aa_sum;
        volatile Dtype *vbb_sum= bb_sum;
        volatile Dtype *vab_sum= ab_sum;
        if (thread_idx < 16) {
            int index = thread_idx + 16;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
        if (thread_idx < 8) {
            int index = thread_idx + 8;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
        if (thread_idx < 4) {
            int index = thread_idx + 4;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
        if (thread_idx < 4) {
            int index = thread_idx + 2;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
        if (thread_idx < 2) {
            int index = thread_idx + 1;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
    }
    if (thread_idx == 0) {
        auto c = aa_sum[0] * bb_sum[0];
        if (c < epsilon) {
            out_data[block_idx] = 0;
        } else {
            out_data[block_idx] = ab_sum[0] / sqrt(c);
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberCosSim<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        CosSimParam<NV>& param) {
   
    CHECK_EQ(inputs.size(), 2) << "CosSim input num need be  2, but is" << inputs.size();
    CHECK_EQ(outputs.size(), 1) << "CosSim input num need be  1, but is" << outputs.size();
    size_t count_0 = inputs[0]->valid_size();
    size_t count_1 = inputs[1]->valid_size();
    CHECK_EQ(count_0, count_1) << "input0 and input1 valid size is not equal";

    size_t num = inputs[0]->num();
    size_t inner_size = count_0 / inputs[0]->num();

    const OpDataType *in_0_data = (const OpDataType*)inputs[0]->data();
    const OpDataType *in_1_data = (const OpDataType*)inputs[1]->data();
    
    OpDataType *out_data = (OpDataType*)outputs[0]->mutable_data();

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    float epsilon = param.epsilon;
    
    int block_size = exp2(floor(log2(float(inner_size))));
    block_size = std::min(block_size, CUDA_NUM_THREADS);
    
    ker_cos_sim_fwd<OpDataType>
            <<<num, block_size, 3*block_size*sizeof(OpDataType), cuda_stream>>>(
            out_data, in_0_data, in_1_data, num, inner_size, epsilon);

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberCosSim<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberCosSim, CosSimParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberCosSim, CosSimParam, NV, AK_HALF);
}
}
