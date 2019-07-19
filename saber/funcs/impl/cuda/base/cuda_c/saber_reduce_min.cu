#include "saber/funcs/impl/cuda/saber_reduce_min.h"

namespace anakin {
namespace saber {

/**
 * @brief reduce tensor acorrding to the given reduce dim.
 *        e.g.
 *            input tensor with shape [5, 2, 10, 4] (rank = 4, how many dimentions does a tensor have.)
 *            and the reduce dim may have the following forms:
 *               1) reduce_dim = None, no reduce dim. It means that reduce all dimentions [default]
 *                  output's shape [1, 1, 1, 1].
 *               2) reduce_dim = x, x is the dimention we want to reduce.
 *                  output's shape:
 *                     x = 0, for example, the shape will be [1, 2, 10, 4] if keep_dim is true, otherwise it will be [2*10*4, 1, 1, 1].
 *                     x = 2, for example, the shape will be [5, 2, 1, 4] if keep_dim is true, otherwise it will be [5*2*4, 1, 1, 1].
 *                     and so on.
 *               3) reduce_dim = [x, y], It will reduce two dimetions x and y.
 *                   output's shape:
 *                     reduce_dim = [0, 1], for example, the shape will be [1, 1, 10 ,4] or [10*4, 1, 1, 1] and so on.
 *               Notes:
 *                  if reduce_dim[i] < 0:
 *                     do 
 *                        reduce_dim[i] += rank.
 * 
 * @tparam OpDtype 
 * @param inputs 
 * @param outputs 
 * @param param 
 * @return SaberStatus 
 */

 //This function is used to implement atioMin based on CAS function.
//  __device__ float atomicMin(float* address, float val) {
//      unsigned long long int* address_as_ull = (unsigned long long int*)address;
//      unsigned long long int old = *address_as_ull, assumed;
//      do{
//          assumed = old;
//          old = atomicCAS(address_as_ull, assumed, __float_as_longlong(
//                                                     fminf(val, __longlong_as_float(assumed))));

//      }while(assumed != old);
//      return __longlong_as_float(old);
//  }

 __device__ double atomicMin(double* address, double val) {
     unsigned long long int* address_as_ull = (unsigned long long int*)address;
     unsigned long long int old = *address_as_ull, assumed;
     do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(
                                                   fmin(val, __longlong_as_double(assumed))));

    }while(assumed != old);
    return __longlong_as_double(old);
 }

 __device__ double atomicMin(float* address, float val) {
     unsigned long long int* address_as_ull = (unsigned long long int*)address;
     unsigned long long int old = *address_as_ull, assumed;
     do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __float_as_int(
                                                   fminf(val, __int_as_float(assumed))));

    }while(assumed != old);
    return __longlong_as_double(old);
 }

//thread num: CHW
template <typename dtype>
__global__ void kernel_reduce_n(const dtype* src, dtype* dst, 
                const int num_in, const int channel_in, const int height_in, const int width_in, const int count) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = blockDim.x * gridDim.x;
    int feature_map = height_in * width_in; //HW
    int size = channel_in * feature_map;// CHW
    int c_id = tid / feature_map; 
    int feature_map_inner_index = tid % feature_map;
    dtype min = src[tid];
    for (int n = 1; n < num_in; ++n) {
        dtype tmp = src[n * size + c_id * feature_map + feature_map_inner_index];
        min = tmp < min ? tmp : min;
    }
    dst[tid] = min;
}

//thread num:NHW
template <typename dtype>
__global__ void kernel_reduce_c(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in, const int count) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = blockDim.x * gridDim.x;
    int feature_map = height_in * width_in;
    int size = channel_in * feature_map;
    for (int i = tid; i < count; i += thread_num) {
        int n_id = i / feature_map;
        int inner_index = i % feature_map;
        dtype min = src[n_id * size + inner_index];
        for (int c = 1; c < channel_in; ++c) {
            dtype tmp = src[n_id * size + c * feature_map + inner_index];
            min = tmp < min? tmp : min;
        }
        dst[n_id * feature_map + inner_index] = min; // Is data_index same to tid/i?.
    }

}

//thread num: NCW
template <typename dtype>
__global__ void kernel_reduce_h(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in, const int count) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = blockDim.x * gridDim.x;
    int feature_map = height_in * width_in; //CW
    int cw_size = channel_in * width_in; //CW
    int size = channel_in * feature_map; //CHW
    for (int i = tid; i < count; i += thread_num) {
        int n_id = i / cw_size;
        int c_id = (i / width_in) % channel_in;
        int inner_index = i % width_in;
        int data_index = n_id * size + c_id * feature_map + inner_index;
        dtype min = src[data_index];
        for (int h = 1; h < height_in; ++h) {
            dtype tmp = src[data_index + h * width_in];
            min = tmp < min? tmp : min;
        }
        dst[n_id * cw_size + c_id * width_in + inner_index] = min; // Is data_index same to tid/i?.
    }
}

//thread num: NCH
template <typename dtype>
__global__ void kernel_reduce_w(const dtype* src, dtype* dst, 
    const int num_in, const int channel_in, const int height_in, const int width_in, const int count) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = blockDim.x * gridDim.x;
    int ch_size = channel_in * height_in; //CH
    int size = ch_size * width_in; //CHW
    int feature_map = height_in * width_in; //HW
    for (int i = tid; i < count; i += thread_num) {
        int n_id = i / ch_size;
        int c_id = (i / height_in) % channel_in;
        int inner_index = i % height_in;
        int data_index = n_id * size + c_id * feature_map + inner_index * width_in;
        dtype min = src[data_index];
        for (int w = 1; w < width_in; ++w) {
            dtype tmp = src[data_index + w];
            min = tmp < min? tmp : min;
        }
        dst[n_id * ch_size + c_id * height_in + inner_index] = min;
    }
}

//reduce all.
template <typename dtype>
__global__ void kernel_reduce_nchw(const dtype* src, dtype* dst, const int count) {

    int n_id = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int thread_num = blockDim.x * gridDim.x;
    dst[0] = src[n_id];
    extern __shared__ dtype s[];
    dtype min = src[n_id];
    for (int i = n_id; i < count; i += thread_num) {
        min = src[i] < min ? src[i] : min;
    }
    s[tid] = min;
    __syncthreads();

    int powOf2 = blockDim.x;
    if (powOf2 & (powOf2 - 1)) {
        //block threads are not pow of 2.
        while (powOf2 & (powOf2 - 1)) {
            powOf2 &= powOf2 - 1;
        } // it'll end when it find pow of 2.
        if (tid >= powOf2) {
            s[tid - powOf2] = s[tid - powOf2] < s[tid]? s[tid - powOf2] : s[tid]; 
        }
        __syncthreads();
    }
    for (int i = powOf2>>1; i > 0; i>>=1) {
        if (tid < i) {
            s[tid] = s[tid] < s[tid + i]? s[tid] : s[tid + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        //double tmp = s[]
        atomicMin(&dst[0], s[threadIdx.x]);
    }
}

template <DataType OpDtype>
SaberStatus SaberReduceMin<NV, OpDtype>::dispatch(const std::vector<Tensor<NV>*>& inputs,
    std::vector<Tensor<NV>*>& outputs,
    ReduceMinParam<NV>& param) {
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    const OpDataType* input_ptr = (const OpDataType*)inputs[0]->data();
    OpDataType* output_ptr = (OpDataType*)outputs[0]->mutable_data();
    int count = outputs[0]->valid_size();

    if (_reduce_dim.empty()) {
        // reduce_all
        int count_all = inputs[0]->valid_size();
        int grid, thread_num;
        if (count_all < CUDA_NUM_THREADS) {
            thread_num = count_all;
            grid = 1;
        }else {
            thread_num = CUDA_NUM_THREADS;
            if (CUDA_GET_BLOCKS(count) >= 128) //This is to avoid share memory blowing up.
                grid = 64;
            else
                grid = CUDA_GET_BLOCKS(count);
        }
        int sharedSize = thread_num * 4;
        kernel_reduce_nchw<OpDataType><<<grid, thread_num, sharedSize, cuda_stream>>>(
            input_ptr, output_ptr, count_all);
    }else if (_reduce_dim.size() == 1) {
        if (_reduce_dim[0] == 0) {
            //reduce n
            kernel_reduce_n<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                input_ptr, output_ptr, _num, _channel, _height, _width, count);
        }
        if (_reduce_dim[0] == 1) {
            //reduce c
            kernel_reduce_c<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                input_ptr, output_ptr, _num, _channel, _height, _width, count);
        }
        if (_reduce_dim[0] == 2) {
            //reduce h
            kernel_reduce_h<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                input_ptr, output_ptr, _num, _channel, _height, _width, count);
        }
        if (_reduce_dim[0] == 3) {
            //reduce h
            kernel_reduce_w<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                input_ptr, output_ptr, _num, _channel, _height, _width, count);
        }
    } else if (_reduce_dim.size() == 2) {
        //only consecutive reduce dim? [0,1] [1, 2], not [0, 2]?
        if (_reduce_dim[0] == 0 && _reduce_dim[1] == 1) {
            //reduce n, c. reduce n first.
            _tensor_tmp.reshape(std::vector<int>({1, _channel, _height, _width}));
            int count_n = _tensor_tmp.valid_size();
            int count_nc = count_n / _tensor_tmp.channel();
            OpDataType* tmp_out = (OpDataType*)_tensor_tmp.mutable_data();
            kernel_reduce_n<OpDataType><<<CUDA_GET_BLOCKS(count_n), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                input_ptr, tmp_out, _num, _channel, _height, _width, count_n);
            
            kernel_reduce_c<OpDataType><<<CUDA_GET_BLOCKS(count_nc), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                tmp_out, output_ptr, 1, _channel, _height, _width, count_nc);
        }else if (_reduce_dim[0] == 1 && _reduce_dim[1] == 2) {
            //reduce c. h. reduce c first.
            _tensor_tmp.reshape(std::vector<int>({_num, 1, _height, _width}));
            int count_c = _tensor_tmp.valid_size();
            int count_ch = count_c / _tensor_tmp.height();
            OpDataType* tmp_out = (OpDataType*)_tensor_tmp.mutable_data();
            kernel_reduce_c<OpDataType><<<CUDA_GET_BLOCKS(count_c), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                input_ptr, tmp_out, _num, _channel, _height, _width, count_c);
            
            kernel_reduce_h<OpDataType><<<CUDA_GET_BLOCKS(count_ch), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                tmp_out, output_ptr, _num, 1, _height, _width, count_ch);
        }else if (_reduce_dim[0] == 2 && _reduce_dim[1] == 3) {
            //reduce h, w. reduce h first.
            _tensor_tmp.reshape(std::vector<int>({_num, _channel, 1, _width}));
            int count_h = _tensor_tmp.valid_size();
            int count_hw = count_h / _tensor_tmp.width();
            OpDataType* tmp_out = (OpDataType*)_tensor_tmp.mutable_data();
            kernel_reduce_h<OpDataType><<<CUDA_GET_BLOCKS(count_h), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                input_ptr, tmp_out, _num, _channel, _height, _width, count_h);

            kernel_reduce_w<OpDataType><<<CUDA_GET_BLOCKS(count_hw), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                tmp_out, output_ptr, _num, _channel, 1, _width, count_hw);
        }else {
            LOG(FATAL) <<"[reduce_min] invalid reduce_dim!!!";
        }
    }else {
        LOG(FATAL) << "[reduce_min]Reducing size over than 2 is not support!!";
    }

    CUDA_POST_KERNEL_CHECK;

    return SaberSuccess;
}

template class SaberReduceMin<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberReduceMin, ReduceMinParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberReduceMin, ReduceMinParam, NV, AK_INT8);

} // namespace saber.
} // namespace anakin.
