#include "saber/funcs/impl/cuda/saber_permute.h"
#include "cuda_fp16.h"

namespace anakin {

namespace saber {

const int TRANS_BLOCK_SIZE = 16;

template <typename Dtype>
__global__ void ker_permute_fwd(Dtype * out_data, const int num_axes,\
                    const int count, const int * permute_order,\
                    const int * new_steps, const int * old_steps,\
                    const Dtype* in_data)
{
    CUDA_KERNEL_LOOP(tid, count){
        int org_idx = tid;
        int in_idx = 0;
        #pragma unroll
        for (int i = 0; i < num_axes; i++) {
            int order = permute_order[i];
            int new_step = new_steps[i];
            int old_step = old_steps[order];
            in_idx += (org_idx / new_step) * old_step;
            org_idx %= new_step;
        }
        out_data[tid] = in_data[in_idx];
    }
}

template <typename Dtype>
__global__ void ker_permute_fwd(Dtype * out_data, const int num_axes,\
                    const int count, const int * permute_order,\
                    const int * new_steps, const int * old_steps,\
                    const int * new_valid_shape,
                    const Dtype* in_data)
{
    CUDA_KERNEL_LOOP(tid, count){
        int in_idx = 0;
        int out_idx  = 0;
        int new_valid_stride = 1;
        #pragma unroll
        for (int i = num_axes - 1; i >= 0; --i) {
            int order = permute_order[i];
            int new_step = new_steps[i];
            int old_step = old_steps[order];
            int id = (tid / new_valid_stride) % new_valid_shape[i];
            in_idx += id * old_step;
            out_idx += id * new_step;
            new_valid_stride *= new_valid_shape[i];
        }
        out_data[out_idx] = in_data[in_idx];
    }
}
/*in this kernel, we suppose img with format (1, h, w, c) tranform to (1, c, h, w), 
and c = 3. out_h = c, out_w = h * w. each thread process one pixel*/
template<typename Dtype>
__global__ void ker_permute_fwd_transpose(Dtype * out_data, \
                    const int out_h, const int out_w, \
                    const Dtype* in_data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float tile[3][CUDA_NUM_THREADS];
    if (tid < out_w) {
        int offset = tid * out_h;
        tile[0][threadIdx.x] = in_data[offset];
        tile[1][threadIdx.x] = in_data[offset + 1];
        tile[2][threadIdx.x] = in_data[offset + 2];
    }
    __syncthreads();
    if (tid < out_w) {
        out_data[0 *out_w + tid] = tile[0][threadIdx.x];
        out_data[1 *out_w + tid] = tile[1][threadIdx.x];
        out_data[2 *out_w + tid] = tile[2][threadIdx.x];
    }
}

template<typename Dtype>
__global__ void ker_permute_fwd_transpose(Dtype * out_data, \
                    const int n,
                    const int c,
                    const int h, 
                    const int w,
                    const int * out_stride,
                    const int * in_stride,
                    const Dtype* in_data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float tile[3][CUDA_NUM_THREADS];
    int out_w_id = tid % w;
    int out_h_id = (tid / w) % h;
    int out_n_id = tid / (h * w);
    int out_offset = out_n_id * out_stride[0] + out_h_id * out_stride[2] + out_w_id * out_stride[3];
    int in_offset = out_n_id * in_stride[0] + out_h_id * in_stride[1] + out_w_id * in_stride[2]; 
    if (tid < n * h * w) {
        tile[0][threadIdx.x] = in_data[in_offset];
        tile[1][threadIdx.x] = in_data[in_offset + 1];
        tile[2][threadIdx.x] = in_data[in_offset + 2];
    }
    __syncthreads();
    if (tid < n * h * w ){
        out_data[out_offset + out_stride[1] * 0] = tile[0][threadIdx.x];
        out_data[out_offset + out_stride[1] * 1] = tile[1][threadIdx.x];
        out_data[out_offset + out_stride[1] * 2] = tile[2][threadIdx.x];
    }
}

/*in this kernel, we suppose img with format (1, c, h, w) tranform to (1, h, w, c), 
and out_h = h*w, out_w = c. each thread process one data. we use share memory*/
template<typename Dtype>
__global__ void ker_transpose(Dtype * out_data, \
                    const int out_h, const int out_w, \
                    const Dtype* in_data)
{
    __shared__ float tile[TRANS_BLOCK_SIZE][TRANS_BLOCK_SIZE];
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;//in index
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;//in index
    if (tid_x < out_h && tid_y < out_w) {
        tile[threadIdx.x][threadIdx.y] = in_data[tid_x + tid_y * out_h];
    }
    __syncthreads();
    if (tid_x < out_h && tid_y < out_w) {
        out_data[tid_x * out_w + tid_y] = tile[threadIdx.x][threadIdx.y];
    }
}

template<typename Dtype>
__global__ void ker_nchw_to_nhwc(Dtype * out_data, 
                    const int n, 
                    const int c, 
                    const int h,
                    const int w, 
                    const int * out_stride,
                    const int * in_stride,
                    const Dtype* in_data)
{
    __shared__ float tile[TRANS_BLOCK_SIZE][TRANS_BLOCK_SIZE];
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;//in index
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;//in index
    int w_id = tid_y % w;
    int h_id = tid_y / w;
    int c_id = tid_x % c;
    int n_id = tid_x / c;
    int in_offset = n_id * in_stride[0] + c_id * in_stride[1] \
             + h_id * in_stride[2] + w_id * in_stride[3];
    int out_offset = n_id * out_stride[0] + h_id * out_stride[1] + \
             w_id * out_stride[2]  + c_id * out_stride[3];
    if (tid_x < n*c && tid_y < h*w) {
        tile[threadIdx.x][threadIdx.y] = in_data[in_offset];
    }
    __syncthreads();
    
    if (tid_x < n*c && tid_y < h*w) {
        out_data[out_offset] = tile[threadIdx.x][threadIdx.y];
    }
}
template <>
SaberStatus SaberPermute<NV, AK_FLOAT>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    PermuteParam<NV>& param) {

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    const float* in_data =static_cast<const float*>(inputs[0]->data());
    float* out_data = static_cast<float*>(outputs[0]->mutable_data());
    int count = outputs[0]->valid_size();
    const int* permute_order = static_cast<const int*>(_permute_order.data());
    const int* new_steps = static_cast<const int*>(_out_steps.data());
    const int* old_steps = static_cast<const int*>(_in_steps.data());
    const int* out_valid_shape = static_cast<const int*>(_out_valid_shape.data());
    std::vector<int> permute_order_nhwc_to_nchw = {0, 3, 1, 2};
    PermuteParam<NV> param_nhwc_to_nchw(permute_order_nhwc_to_nchw);
    std::vector<int> permute_order_nchw_to_nhwc = {0, 2, 3, 1};
    PermuteParam<NV> param_nchw_to_nhwc(permute_order_nchw_to_nhwc);
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        if (_need_permute) {
            if (inputs[0]->num() == 1 && inputs[0]->width() == 3
                && param == param_nhwc_to_nchw) {
                int out_w = outputs[0]->width() * outputs[0]->height();
                int out_h = outputs[0]->channel();
                ker_permute_fwd_transpose<float>\
                        <<<CUDA_GET_BLOCKS(out_w), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, out_h, out_w, in_data);
            } else if (inputs[0]->num() == 1 && param == param_nchw_to_nhwc) {
                int out_h = inputs[0]->width() * inputs[0]->height();
                int out_w = inputs[0]->channel();
                dim3 block_size(TRANS_BLOCK_SIZE, TRANS_BLOCK_SIZE);
                dim3 grid_size((out_h + TRANS_BLOCK_SIZE - 1) / TRANS_BLOCK_SIZE, (out_w + TRANS_BLOCK_SIZE - 1) / TRANS_BLOCK_SIZE);
                ker_transpose<float>\
                        <<<grid_size, block_size, 0, cuda_stream>>>(\
                        out_data, out_h, out_w, in_data);
            } else {
                ker_permute_fwd<float>\
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, _num_axes, count, permute_order, \
                        new_steps, old_steps, in_data);
            }
        } else {
               outputs[0]->copy_from(*inputs[0]);
            //outputs[0]->share_from(inputs[0]);
        }
    } else {
        if (_need_permute) {
            if (inputs[0]->num() == 1 && inputs[0]->width() == 3 
                && param == param_nhwc_to_nchw) {
                int out_w = outputs[0]->width() * outputs[0]->height();
                int out_h = outputs[0]->channel();
                ker_permute_fwd_transpose<float>\
                        <<<CUDA_GET_BLOCKS(out_w), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, outputs[0]->num(), outputs[0]->channel(), \
                        outputs[0]->height(), outputs[0]->width(),
                        new_steps, old_steps, in_data);
            } else if (inputs[0]->num() == 1 && param == param_nchw_to_nhwc) {
                dim3 block_size(TRANS_BLOCK_SIZE, TRANS_BLOCK_SIZE);
                dim3 grid_size((inputs[0]->num() * inputs[0]->channel() + TRANS_BLOCK_SIZE - 1) / TRANS_BLOCK_SIZE,
                        (inputs[0]->height() * inputs[0]->width() + TRANS_BLOCK_SIZE - 1) / TRANS_BLOCK_SIZE);
                ker_nchw_to_nhwc<float>\
                        <<<grid_size, block_size, 0, cuda_stream>>>(\
                        out_data, inputs[0]->num(), inputs[0]->channel(),\
                        inputs[0]->height(), inputs[0]->width(),\
                        new_steps, old_steps, in_data);
            } else {
                ker_permute_fwd<float>\
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, _num_axes, count, permute_order, \
                        new_steps, old_steps, in_data);
            }
        } else {
               outputs[0]->copy_from(*inputs[0]);
            //outputs[0]->share_from(inputs[0]);
        }
    }
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberPermute, PermuteParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberPermute, PermuteParam, NV, AK_INT8);
}
}
