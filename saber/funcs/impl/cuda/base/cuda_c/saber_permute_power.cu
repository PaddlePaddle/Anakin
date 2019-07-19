#include "saber/funcs/impl/cuda/saber_permute_power.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_permute_power_fwd(Dtype * out_data, const int num_axes,\
                    const int count, const int * permute_order,\
                    const int * new_steps, const int * old_steps,\
                    const float scale, const float shift, const float power,\
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
        out_data[tid] = pow(scale * in_data[in_idx] + shift, power);
    }
}

template<typename Dtype>
__global__ void ker_permute_power_fwd_transpose(Dtype * out_data, \
                    const int out_h, const int out_w, \
                    const float scale, const float shift, const float power,\
                    const Dtype* in_data)
{
    __shared__ float tile[3][CUDA_NUM_THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < out_w) {
        tile[0][threadIdx.x] = pow(in_data[tid * out_h + 0] * scale + shift, power);
        tile[1][threadIdx.x] = pow(in_data[tid * out_h + 1] * scale + shift, power);
        tile[2][threadIdx.x] = pow(in_data[tid * out_h + 2] * scale + shift, power);
    }
    __syncthreads();
    if (tid < out_w) {
        out_data[0 *out_w + tid] = tile[0][threadIdx.x];
        out_data[1 *out_w + tid] = tile[1][threadIdx.x];
        out_data[2 *out_w + tid] = tile[2][threadIdx.x];
    }
}

template <typename Dtype>
__global__ void ker_permute_scale_fwd(Dtype * out_data, const int num_axes,\
                    const int count, const int * permute_order,\
                    const int * new_steps, const int * old_steps,\
                    const float scale, const float shift,\
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
        out_data[tid] = scale * in_data[in_idx] + shift;
    }
}

template<typename Dtype>
__global__ void ker_permute_scale_fwd_transpose(Dtype * out_data, \
                    const int out_h, const int out_w, \
                    const float scale, const float shift, \
                    const Dtype* in_data)
{
    __shared__ float tile[3][CUDA_NUM_THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < out_w) {
        tile[0][threadIdx.x] = in_data[tid * out_h + 0] * scale + shift;
        tile[1][threadIdx.x] = in_data[tid * out_h + 1] * scale + shift;
        tile[2][threadIdx.x] = in_data[tid * out_h + 2] * scale + shift;
    }
    __syncthreads();
    if (tid < out_w) {
        out_data[0 *out_w + tid] = tile[0][threadIdx.x];
        out_data[1 *out_w + tid] = tile[1][threadIdx.x];
        out_data[2 *out_w + tid] = tile[2][threadIdx.x];
    }
}

template<typename Dtype>
__global__ void ker_nhwc_to_nchw_scale(Dtype * out_data, \
                    const int n, const int c, \
                    const int h, const int w, \
                    const int out_stride_n, const int out_stride_c,\
                    const int out_stride_h, const int out_stride_w,\
                    const int in_stride_n, const int in_stride_c,\
                    const int in_stride_h, const int in_stride_w,\
                    const float scale, const float shift, \
                    const Dtype* in_data)
{
    __shared__ float tile[3][CUDA_NUM_THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int w_id = tid % w;
    int h_id = (tid / w) % h;
    int n_id = (tid / (h*w)) % n;
    int in_offset = n_id * in_stride_n + h_id * in_stride_h + w_id * in_stride_w;
    int out_offset = n_id * out_stride_n + h_id * out_stride_h + w_id * out_stride_w;
    if (tid < n * h * w) {
        tile[0][threadIdx.x] = in_data[in_offset + 0] * scale + shift;
        tile[1][threadIdx.x] = in_data[in_offset + 1] * scale + shift;
        tile[2][threadIdx.x] = in_data[in_offset + 2] * scale + shift;
    }
    __syncthreads();
    if (tid < n*h*w) {
        out_data[0 * out_stride_c + out_offset] = tile[0][threadIdx.x];
        out_data[1 * out_stride_c + out_offset] = tile[1][threadIdx.x];
        out_data[2 * out_stride_c + out_offset] = tile[2][threadIdx.x];
    }
}

template<typename Dtype>
__global__ void ker_nhwc_to_nchw_power(Dtype * out_data, \
                    const int n, const int c, \
                    const int h, const int w, \
                    const int out_stride_n, const int out_stride_c,\
                    const int out_stride_h, const int out_stride_w,\
                    const int in_stride_n, const int in_stride_c,\
                    const int in_stride_h, const int in_stride_w,\
                    const float scale, const float shift, const float power,\
                    const Dtype* in_data)
{
    __shared__ float tile[3][CUDA_NUM_THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int w_id = tid % w;
    int h_id = (tid / w) % h;
    int n_id = (tid / (h*w)) % n;
    int in_offset = n_id * in_stride_n + h_id * in_stride_h + w_id * in_stride_w;
    int out_offset = n_id * out_stride_n + h_id * out_stride_h + w_id * out_stride_w;
    if (tid < n*h*w) {
        tile[0][threadIdx.x] = pow(in_data[in_offset + 0] * scale + shift, power);
        tile[1][threadIdx.x] = pow(in_data[in_offset + 1] * scale + shift, power);
        tile[2][threadIdx.x] = pow(in_data[in_offset + 2] * scale + shift, power);
    }
    __syncthreads();
    if (tid < n*h*w) {
        out_data[0 * out_stride_c + out_offset] = tile[0][threadIdx.x];
        out_data[1 * out_stride_c + out_offset] = tile[1][threadIdx.x];
        out_data[2 * out_stride_c + out_offset] = tile[2][threadIdx.x];
    }
}

template <typename Dtype>
__global__ void ker_permute_power_fwd(Dtype * out_data, const int num_axes,\
                    const int count, const int * permute_order,\
                    const int * new_steps, const int * old_steps,\
                    const int * new_valid_shape,\
                    const float scale, const float shift, const float power,\
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
        out_data[out_idx] = pow(in_data[in_idx] * scale + shift, power);
    }
}

template <typename Dtype>
__global__ void ker_permute_scale_fwd(Dtype * out_data, const int num_axes,\
                    const int count, const int * permute_order,\
                    const int * new_steps, const int * old_steps,\
                    const int * new_valid_shape,\
                    const float scale, const float shift, \
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
        out_data[out_idx] = in_data[in_idx] * scale + shift;
    }
}

template <>
SaberStatus SaberPermutePower<NV, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<NV> *>& inputs, \
                  std::vector<Tensor<NV> *>& outputs, \
                  PermutePowerParam<NV>& param) {

    const float *in_data = static_cast<const float*>(inputs[0]->data());
    float *out_data = static_cast<float*>(outputs[0]->mutable_data());
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    const int * permute_order = static_cast<const int*>(_permute_order.data());
    const int * new_steps = static_cast<const int*>(_new_steps.data());
    const int * old_steps = static_cast<const int*>(_old_steps.data());
    const int * new_valid_shape = static_cast<const int*>(_out_valid_shape.data());
    const float scale = param.has_power_param ? param.power_param.scale : 1.0f;
    const float shift = param.has_power_param ? param.power_param.shift : 0.0f;
    const float power = param.has_power_param ? param.power_param.power : 1.0f;
    std::vector<int> permute_order_t = {0, 3, 1, 2};
    PermuteParam<NV> param_t(permute_order_t);
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        if (inputs[0]->num() == 1 && inputs[0]->width() == 3 
            && param.permute_param == param_t && 1) {
            int out_w = outputs[0]->width() * outputs[0]->height();
            int out_h = outputs[0]->channel();
            if (power != 1.0f) {
                ker_permute_power_fwd_transpose<float>\
                        <<<CUDA_GET_BLOCKS(out_w), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                         out_data, out_h, out_w, scale, shift, power, in_data);
            } else {
                ker_permute_scale_fwd_transpose<float>\
                        <<<CUDA_GET_BLOCKS(out_w), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                         out_data, out_h, out_w, scale, shift, in_data);
            }
        } else {
            if (power != 1.0f) {
                ker_permute_power_fwd<float>\
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, _num_axes, count, permute_order, \
                        new_steps, old_steps, 
                        scale, shift, power, 
                        in_data);
            } else {
                ker_permute_scale_fwd<float>\
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, _num_axes, count, permute_order, \
                        new_steps, old_steps, 
                        scale, shift, 
                        in_data);
            }
        }
    } else {
        if (inputs[0]->width() == 3 && param.permute_param == param_t) {
            const int out_n = outputs[0]->num();
            const int out_c = outputs[0]->channel();
            const int out_h = outputs[0]->height();
            const int out_w = outputs[0]->width();
            const int count = out_n * out_h * out_w;
            Shape out_stride = outputs[0]->get_stride();
            Shape in_stride = inputs[0]->get_stride();
            if (power != 1.0f) {
                ker_nhwc_to_nchw_power<float>
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, \
                        out_n, out_c, \
                        out_h, out_w, \
                        out_stride[0], out_stride[1],\
                        out_stride[2], out_stride[3],\
                        in_stride[0], in_stride[3],\
                        in_stride[1], in_stride[2],\
                        scale, shift, power,\
                        in_data);
            } else {
                ker_nhwc_to_nchw_scale<float>
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, \
                        out_n, out_c, \
                        out_h, out_w, \
                        out_stride[0], out_stride[1],\
                        out_stride[2], out_stride[3],\
                        in_stride[0], in_stride[3],\
                        in_stride[1], in_stride[2],\
                        scale, shift, \
                        in_data);
            }
        } else {
            int count = outputs[0]->valid_size();
            if (power != 1.0f) {
                ker_permute_power_fwd<float>\
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, _num_axes, count, permute_order, \
                        new_steps, old_steps,
                        new_valid_shape,
                        scale, shift, power, 
                        in_data);
            } else {
                ker_permute_scale_fwd<float>\
                        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, _num_axes, count, permute_order, \
                        new_steps, old_steps, 
                        new_valid_shape,
                        scale, shift, 
                        in_data);
            }
            
        }
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberPermutePower, PermutePowerParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberPermutePower, PermutePowerParam, NV, AK_INT8);
}
}
