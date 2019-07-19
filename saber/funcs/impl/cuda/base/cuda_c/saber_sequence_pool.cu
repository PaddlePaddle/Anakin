#include "core/common.h"
#include "saber/funcs/impl/cuda/saber_sequence_pool.h"
#include "saber/saber_funcs_param.h"
#include "cuda.h"
namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void seq_pool_average_kernel(Dtype* dst, const Dtype* src_in,const int batch_size,
                             const int* seq_offset, const int slice_size){
    int total = slice_size * batch_size;
    CUDA_KERNEL_LOOP(tid, total){
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_slice_num = seq_offset[out_batch_id + 1] - seq_offset[out_batch_id];
        int in_offset = seq_offset[out_batch_id] * slice_size;
        src_in += in_offset + out_id;
        Dtype sum = (Dtype)0;
        for(int i = 0; i < in_slice_num; ++i){
            sum += src_in[i * slice_size];
        }
        dst[out_batch_id * slice_size + out_id] = sum / in_slice_num; 
    }
}

template <typename Dtype>
__global__ void seq_pool_sum_kernel(Dtype* dst, const Dtype* src_in,const int batch_size,
                             const int* seq_offset, const int slice_size){
    int total = slice_size * batch_size;
    CUDA_KERNEL_LOOP(tid, total){
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_slice_num = seq_offset[out_batch_id + 1] - seq_offset[out_batch_id];
        int in_offset = seq_offset[out_batch_id] * slice_size;
        src_in += in_offset + out_id;
        Dtype sum = (Dtype)0;
        for(int i = 0; i < in_slice_num; ++i){
            sum += src_in[i * slice_size];
        }
        dst[out_batch_id * slice_size + out_id] = sum; 
    }
}

template <typename Dtype>
__global__ void seq_pool_sqrt_kernel(Dtype* dst, const Dtype* src_in,const int batch_size,
                             const int* seq_offset, const int slice_size){
    int total = slice_size * batch_size;
    CUDA_KERNEL_LOOP(tid, total){
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_slice_num = seq_offset[out_batch_id + 1] - seq_offset[out_batch_id];
        int in_offset = seq_offset[out_batch_id] * slice_size;
        src_in += in_offset + out_id;
        Dtype sum = (Dtype)0;
        for(int i = 0; i < in_slice_num; ++i){
            sum += src_in[i * slice_size];
        }
        dst[out_batch_id * slice_size + out_id] = sum * rsqrtf(in_slice_num); 
    }
}

template <typename Dtype>
__global__ void seq_pool_max_kernel(Dtype* dst, const Dtype* src_in,const int batch_size,
                             const int* seq_offset, const int slice_size){
    int total = slice_size * batch_size;
    CUDA_KERNEL_LOOP(tid, total){
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_slice_num = seq_offset[out_batch_id + 1] - seq_offset[out_batch_id];
        int in_offset = seq_offset[out_batch_id] * slice_size;
        src_in += in_offset + out_id;
        Dtype max = src_in[0];
        for (int i = 1; i < in_slice_num; ++i){
            Dtype val = src_in[i * slice_size];
            if (val > max){
                max = val;
            }
        }
        dst[out_batch_id * slice_size + out_id] = max; 
    }
}

template <typename Dtype>
__global__ void seq_pool_last_kernel(Dtype* dst, const Dtype* src_in, const int batch_size,
                             const int* seq_offset, const int slice_size) {
    int total = slice_size * batch_size;
    CUDA_KERNEL_LOOP(tid, total){
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_offset = (seq_offset[out_batch_id + 1]  - 1) * slice_size;
        dst[tid] = src_in[in_offset + out_id];
    }
}

template <typename Dtype>
__global__ void seq_pool_first_kernel(Dtype* dst, const Dtype* src_in, const int batch_size,
                             const int* seq_offset, const int slice_size) {
    int total = slice_size * batch_size;
    CUDA_KERNEL_LOOP(tid, total){
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_offset = seq_offset[out_batch_id] * slice_size;
        dst[tid] = src_in[in_offset + out_id];
    }
}



template <typename dtype>
void seq_pool_average(dtype* dst, const dtype* src_in,const int batch_size,
                      const int* seq_offset, const int slice_size, Context<NV>* ctx) {
    seq_pool_average_kernel<<<CUDA_GET_BLOCKS(batch_size * slice_size),CUDA_NUM_THREADS, 0, ctx->get_compute_stream()>>>\
        (dst,src_in,batch_size,seq_offset,slice_size);
}

template <typename dtype>
void seq_pool_sum(dtype* dst, const dtype* src_in,const int batch_size,
                      const int* seq_offset, const int slice_size, Context<NV>* ctx) {
    seq_pool_sum_kernel<<<CUDA_GET_BLOCKS(batch_size * slice_size),CUDA_NUM_THREADS, 0, ctx->get_compute_stream()>>>\
        (dst,src_in,batch_size,seq_offset,slice_size);
}

template <typename dtype>
void seq_pool_sqrt(dtype* dst, const dtype* src_in,const int batch_size,
                      const int* seq_offset, const int slice_size, Context<NV>* ctx) {
    seq_pool_sqrt_kernel<<<CUDA_GET_BLOCKS(batch_size * slice_size),CUDA_NUM_THREADS, 0, ctx->get_compute_stream()>>>\
        (dst,src_in,batch_size,seq_offset,slice_size);
}

template <typename dtype>
void seq_pool_max(dtype* dst, const dtype* src_in,const int batch_size,
                      const int* seq_offset, const int slice_size, Context<NV>* ctx) {
    seq_pool_max_kernel<<<CUDA_GET_BLOCKS(batch_size * slice_size),CUDA_NUM_THREADS, 0, ctx->get_compute_stream()>>>\
        (dst,src_in,batch_size,seq_offset,slice_size);
}

template <typename dtype>
void seq_pool_first(dtype* dst, const dtype* src_in,const int batch_size,
                      const int* seq_offset, const int slice_size, Context<NV>* ctx) {
    seq_pool_first_kernel<<<CUDA_GET_BLOCKS(batch_size * slice_size),CUDA_NUM_THREADS, 0, ctx->get_compute_stream()>>>\
        (dst,src_in,batch_size,seq_offset,slice_size);
}

template <typename dtype>
void seq_pool_last(dtype* dst, const dtype* src_in,const int batch_size,
                      const int* seq_offset, const int slice_size, Context<NV>* ctx) {
    seq_pool_last_kernel<<<CUDA_GET_BLOCKS(batch_size * slice_size),CUDA_NUM_THREADS, 0, ctx->get_compute_stream()>>>\
        (dst,src_in,batch_size,seq_offset,slice_size);
    
}

template <typename dtype>
void seq_pool_unknow(dtype* dst, const dtype* src_in,const int batch_size,
                      const int* seq_offset, const int slice_size, Context<NV>* ctx) {
    LOG(ERROR) << " UNKNOWN seq pool type";
}

template <DataType OpDtype>
SaberStatus SaberSequencePool<NV, OpDtype>::init(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SequencePoolParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    kernel_direct_map = {
        {Sequence_pool_unknow, seq_pool_unknow<DataType_in>},
        {Sequence_pool_average, seq_pool_average<DataType_in>},
        {Sequence_pool_sum, seq_pool_sum<DataType_in>},
        {Sequence_pool_sqrt, seq_pool_sqrt<DataType_in>},
        {Sequence_pool_max, seq_pool_max<DataType_in>},
        {Sequence_pool_last, seq_pool_last<DataType_in>},
        {Sequence_pool_first, seq_pool_first<DataType_in>},
    };
    return create(inputs, outputs, param, ctx);

}

template <DataType OpDtype>
SaberStatus SaberSequencePool<NV, OpDtype>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SequencePoolParam<NV>& param) {

    CHECK_EQ(inputs[0]->channel(), outputs[0]->channel());
    CHECK_EQ(inputs[0]->height(), outputs[0]->height());
    CHECK_EQ(inputs[0]->width(), outputs[0]->width());

    std::vector<int> seq_offset = inputs[0]->get_seq_offset()[0];
    int slice_size = outputs[0]->channel()
                     * outputs[0]->height()
                     * outputs[0]->width();
    DataType_in* dst_ptr = (DataType_in*)outputs[0]->mutable_data();
    const DataType_out* src_ptr = (const DataType_out*)inputs[0]->data();
    int batch_size = seq_offset.size()-1;
    Tensor<NV> seq_offset_D;
    seq_offset_D.re_alloc(Shape({1, 1, 1, (int)seq_offset.size()}), AK_INT32);
    CUDA_CHECK(cudaMemcpyAsync(seq_offset_D.mutable_data(), seq_offset.data(), \
        sizeof(int) * seq_offset.size(),cudaMemcpyHostToDevice,this->_ctx->get_compute_stream()));
    kernel_direct_map[param.sequence_pool_type](dst_ptr, src_ptr, batch_size, (const int*)seq_offset_D.data(), slice_size, this->_ctx);

    std::vector<int> offset_new(batch_size + 1);

    for (int i = 0; i <= batch_size; ++i) {
        offset_new[i] = i;
    }
    std::vector<std::vector<int>> voffset_new;
    voffset_new.push_back(offset_new);
    outputs[0]->set_seq_offset(voffset_new);
    return SaberSuccess;

}

template class SaberSequencePool<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSequencePool, SequencePoolParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequencePool, SequencePoolParam, NV, AK_INT8);
}
} // namespace anakin
