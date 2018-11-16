#include "saber/funcs/impl/cuda/saber_im2sequence.h"
#include "cuda_fp16.h"

namespace anakin {

namespace saber {

template <typename Dtype>
__global__ void ker_im2sequence_fwd(Dtype * out_data, \
                    const Dtype* in_data,
                    const int in_n,
                    const int in_c,
                    const int in_h,
                    const int in_w,
                    const int out_h,
                    const int out_w,
                    const int col_height,
                    const int col_width,
                    const int window_h,
                    const int window_w,
                    const int pad_up,
                    const int pad_left,
                    const int stride_h,
                    const int stride_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int kernel_exten_h,
                    const int kernel_exten_w,
                    const int num_threads)
{
    CUDA_KERNEL_LOOP(tid, num_threads){
        int out_size_per_img = out_h * out_w;
        int w = tid % out_w;
        int h = (tid / out_w) % out_h;
        int n = (tid / out_size_per_img) % in_n;
        int c = tid / (out_size_per_img * in_n);
        
        int in_start_w = w * stride_w - pad_left;
        int in_start_h = h * stride_h - pad_up;
        int in_end_w = in_start_w + kernel_exten_w;
        int in_end_h = in_start_h + kernel_exten_h;
        int in_offset = (n * in_c + c) * in_h * in_w;
        const Dtype* in_data_tmp = in_data + in_offset;
        int out_offset = (tid % col_height  * in_c  +  c ) *  window_h * window_w;
        Dtype* out_data_tmp = out_data;
       
        for (int i = in_start_h; i < in_end_h; i += dilation_h) {
            for (int j = in_start_w; j < in_end_w; j += dilation_w) {
                 if (i < 0 || i >= in_h || j < 0 || j >= in_w) {
                     out_data_tmp[out_offset++] = 0;
                 } else {
                     out_data_tmp[out_offset++] = in_data_tmp[i * in_w + j];
                 }
            }
        }
    }
}

template <typename Dtype>
__global__ void ker_im2sequence_fwd_shared(Dtype * out_data, \
                    const Dtype* in_data,
                    const int in_n,
                    const int in_c,
                    const int in_h,
                    const int in_w,
                    const int out_h,
                    const int out_w,
                    const int col_height,
                    const int col_width,
                    const int window_h,
                    const int window_w,
                    const int pad_up,
                    const int pad_left,
                    const int stride_h,
                    const int stride_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int kernel_exten_h,
                    const int kernel_exten_w,
                    const int num_threads)
{
        int thread_id = threadIdx.x;
        int tid = thread_id + blockIdx.x * blockDim.x;
        extern  __shared__ Dtype share_data[];
        int out_size_per_img = out_h * out_w;
        int w = tid % out_w;
        int h = (tid / out_w) % out_h;
        int n = (tid / out_size_per_img) % in_n;
        int c = tid / col_height;
        
        int in_start_w = w * stride_w - pad_left;
        int in_start_h = h * stride_h - pad_up;
        int in_end_w = in_start_w + kernel_exten_w;
        int in_end_h = in_start_h + kernel_exten_h;
        int in_offset = (n * in_c + c) * in_h * in_w;
        const Dtype* in_data_tmp = in_data + in_offset;
        int window_size = window_h * window_w;
       
        int id = 0;
        for (int i = in_start_h; i < in_end_h; i += dilation_h) {
            for (int j = in_start_w; j < in_end_w; j += dilation_w) {
                 Dtype data = 0;
                 if (i < 0 || i >= in_h || j < 0 || j >= in_w) {
                     data = 0;
                     //share_data[id * blockDim.x + thread_id] = 0;
                     //out_data_tmp[out_offset++] = 0;
                 } else {
                     data = in_data_tmp[i * in_w + j];
                     //out_data_tmp[out_offset++] = in_data_tmp[i * in_w + j]
                 }
                 share_data[thread_id * window_size + id] = data;
                 id++;
            }
        }
        __syncthreads();
        int valid_height = fminf(num_threads - blockIdx.x * blockDim.x, blockDim.x);
        //if (threadIdx.x == 0) {
        //     printf("share memory\n");
        //     for (int i = 0; i < valid_height; i++) {
        //          for (int j = 0; j < window_h * window_w; j++) {
        //              printf("%f, ", share_data[i *  window_h * window_w + j]);
        //          }
        //          printf("\n");
        //     }
        //}
        
        for (int i = threadIdx.x; i < valid_height * window_h * window_w; i+=blockDim.x) {
             int h_id = i / window_size;
             int w_id = i % window_size;
             int id = blockIdx.x * blockDim.x + h_id;
             int row_id = id % col_height;
             int col_id = id / col_height * window_size + w_id;
             out_data[row_id * col_width + col_id] = share_data[i];
        }
            
}

template <DataType OpDtype>
SaberStatus SaberIm2Sequence<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    Im2SequenceParam<NV>& param) {

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int n = inputs[0]->num();
    int c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int num_threads = out_n * c;
    std::vector<int>offset(n+1);
    std::vector<std::vector<int>> seq_offset;
    seq_offset.push_back(offset);
    int per_seq_len = out_n / n;
    for (int i = 0; i < n; i++) {
        seq_offset[0].push_back(i * per_seq_len);
    }
    seq_offset[0].push_back(n * per_seq_len);
    outputs[0]->set_seq_offset(seq_offset);
    
//    LOG(INFO)<<"im2sequence out shape"<<"  n: " \
    << outputs[0]->num()<<" c: "<<outputs[0]->channel()<<" h:"<<outputs[0]->height()<<" w:"<<outputs[0]->width();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        if (0) {
            ker_im2sequence_fwd<OpDataType>\
                     <<<CUDA_GET_BLOCKS(num_threads), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                     out_data, in_data, \
                     n, c, in_h, in_w,\
                     _output_height, _output_width,\
                     out_n, out_c,
                     param.window_h, param.window_w,
                     param.pad_up, param.pad_down,
                     param.stride_h, param.stride_w,
                     param.dilation_h, param.dilation_w,
                     _kernel_exten_h, _kernel_exten_w,
                     num_threads);
        } else {
            ker_im2sequence_fwd_shared<OpDataType>\
                     <<<CUDA_GET_BLOCKS(num_threads), CUDA_NUM_THREADS, sizeof(OpDataType)*CUDA_NUM_THREADS * param.window_h* param.window_w, cuda_stream>>>(\
                     out_data, in_data, \
                     n, c, in_h, in_w,\
                     _output_height, _output_width,\
                     out_n, out_c,
                     param.window_h, param.window_w,
                     param.pad_up, param.pad_down,
                     param.stride_h, param.stride_w,
                     param.dilation_h, param.dilation_w,
                     _kernel_exten_h, _kernel_exten_w,
                     num_threads);
        }
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberIm2Sequence, Im2SequenceParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberIm2Sequence, Im2SequenceParam, NV, AK_INT8);
}
}
