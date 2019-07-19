#include <limits>
#include "saber/funcs/impl/cuda/saber_softmax.h"

namespace anakin{

namespace saber{

//! general kernel for softmax
template <typename dtype>
__global__ void softmax_max_kernel(int total_size, const dtype* in_data, dtype* out_data, \
        dtype min_data, int inner_num, int outer_num, int axis_size){

    //! compute data index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int idx_inner = idx % inner_num;
        int idx_outer = (idx / inner_num) * axis_size;
        int real_index = idx_outer * inner_num + idx_inner;
        //! get maximum data across softmax axis
        dtype max_data = min_data;
        for (int i = 0; i < axis_size; ++i) {
            max_data = in_data[real_index] > max_data? in_data[real_index] : max_data;
            real_index += inner_num;
    }
        out_data[idx] = max_data;
    }
}

template <typename dtype>
__global__ void softmax_max_roi_kernel(int total_size, const dtype* in_data, \
        dtype* out_data, dtype min_data, \
        const int* input_stride_real, const int* output_stride_real, const int* shape_valid, \
        int softmax_axis, int axis_size, int dims){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {

        //! compute real data index
        int input_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            if (i == softmax_axis) {
                continue;
            } else {
                int x = idx % shape_valid[i];
                input_real_index += x * input_stride_real[i];
                idx = idx / shape_valid[i];
            }
        }

        //! get maximum data across softmax axis
        dtype max_data = min_data;
        for (int i = 0; i < axis_size; ++i) {
            max_data = in_data[input_real_index] > max_data? \
                    in_data[input_real_index] : max_data;
            input_real_index += i * input_stride_real[softmax_axis];
        }
        out_data[idx] = max_data;
    }
}

template <typename dtype>
__global__ void softmax_sub_exp_sum_kernel(int total_size, const dtype* in_data, \
        dtype* out_data, const dtype* max_data, dtype* sum_data, \
        int inner_num, int outer_num, int axis_size){

    //! compute data index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_size) {
        int idx_inner = idx % inner_num;
        int idx_outer = (idx / inner_num) * axis_size;

        dtype max_data_cur = max_data[idx];
        //dtype *sum_data_cur = &sum_data[idx];
        dtype sum_data_cur = 0;
        int real_index = idx_outer * inner_num + idx_inner;
        //! compute exp and summarize across the softmax axis
        for (int i = 0; i < axis_size; ++i) {
            dtype sub_data = in_data[real_index] - max_data_cur;
            sub_data = expf(sub_data);
            sum_data_cur += sub_data;
            out_data[real_index] = sub_data;
            real_index += inner_num;
        }
        sum_data[idx] = sum_data_cur;
    }
}

template <typename dtype>
__global__ void softmax_sub_exp_sum_roi_kernel(int total_size, \
        const dtype* in_data, dtype* out_data, \
        const dtype* max_data, dtype* sum_data, \
        const int* input_stride_real, const int* output_stride_real, const int* shape_valid, \
        int softmax_axis, int axis_size, int dims){

    //! compute data index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        //! compute real data index
        int output_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            if (i == softmax_axis) {
                continue;
            } else {
                int x = idx % shape_valid[i];
                output_real_index += x * output_stride_real[i];
                idx = idx / shape_valid[i];
            }
        }

        dtype max_data_cur = max_data[idx];
        //dtype *sum_data_cur = &sum_data[idx];
        dtype sum_data_cur = 0;
        //! compute exp and summarize across the softmax axis
        for (int i = 0; i < axis_size; ++i) {
            dtype sub_data = in_data[output_real_index] - max_data_cur;
            sub_data = expf(sub_data);
            sum_data_cur += sub_data;
            out_data[output_real_index] = sub_data;
            output_real_index += output_stride_real[softmax_axis];
        }
        sum_data[idx] = sum_data_cur;
    }
}

template <typename dtype>
__global__ void softmax_divid_output_kernel(int total_size, dtype* io_data, \
        const dtype* sum_data, int inner_num, int outer_num, int axis_size){
    //! compute data index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int idx_inner = idx % inner_num;
        int idx_outer = (idx / inner_num) * axis_size;
        dtype sum_data_cur = sum_data[idx];
        int real_index = idx_outer * inner_num + idx_inner;
        //! compute final result
        for (int i = 0; i < axis_size; ++i) {
            io_data[real_index] = io_data[real_index] / sum_data_cur;
            real_index += inner_num;
        }
    }
}

template <typename dtype>
__global__ void softmax_divid_output_roi_kernel(int total_size, \
        dtype* io_data, const dtype* sum_data, \
        const int* input_stride_real, const int* output_stride_real, const int* shape_valid, \
        int softmax_axis, int axis_size, int dims){
    //! compute data index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        //! compute real data index
        int output_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            if (i == softmax_axis) {
                continue;
            } else {
                int x = idx % shape_valid[i];
                output_real_index += x * output_stride_real[i];
                idx = idx / shape_valid[i];
            }
        }

        dtype sum_data_cur = sum_data[idx];
        //! compute final result
        for (int i = 0; i < axis_size; ++i) {
            io_data[output_real_index] = io_data[output_real_index] / sum_data_cur;
            output_real_index += output_stride_real[softmax_axis];
        }
    }
}

extern __shared__ char tile[];
template <typename dtype>
__global__ void sharemem_softmax_kernel(int total_size, \
        const dtype* in_data, dtype* out_data, \
        int inner_num, int outer_num, int axis_size){

    //__shared__ dtype data[MAX_AXIS_SIZE][CUDA_NUM_THREADS];
    dtype* data = (dtype*)tile + threadIdx.x;

    //! compute thread index and real data index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_size) {
        int idx_inner = idx % inner_num;
        int idx_outer = (idx / inner_num) * axis_size;
        int blocksize = blockDim.x;

        int real_index = idx_outer * inner_num + idx_inner;
        int loop_idx = real_index;
        //! read all data to sharemem in softmax channel
        #pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            data[i * blocksize] = in_data[loop_idx];
            loop_idx += inner_num;
        }

        //! get maximum value in softmax channel
        dtype max_data = data[0];
        #pragma unroll
        for (int i = 1; i < axis_size; ++i) {
            dtype dt = data[i * blocksize];
            if (max_data < dt){
                max_data = dt;
            }
        }

        //! subtract then summarize
        dtype sum = 0;
        #pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            //dtype *dt = &data[i][thread_idx];
            dtype *dt = data + i * blocksize;
            *dt = expf(*dt - max_data);
            sum += *dt;
        }

        //! write back result
        loop_idx = real_index;
        #pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            out_data[loop_idx] = data[i * blocksize] / sum;
            loop_idx += inner_num;
        }
    }
}

template <typename dtype>
__global__ void sharemem_softmax_roi_kernel(int total_size, \
        const dtype* in_data, dtype* out_data, \
        const int* input_stride_real, const int* output_stride_real, const int* shape_valid, \
        int softmax_axis, int axis_size, int dims){

    //__shared__ dtype data[MAX_AXIS_SIZE][CUDA_NUM_THREADS];
    dtype* data = (dtype*)tile + threadIdx.x;

    //! compute thread index and real data index
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = idx1;

    if (idx < total_size) {

        int blocksize = blockDim.x;

        //! compute real data index
        int input_real_index = 0;
        int output_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            if (i == softmax_axis) {
                continue;
            } else {
                int x = idx % shape_valid[i];
                input_real_index += x * input_stride_real[i];
                output_real_index += x * output_stride_real[i];
                idx = idx / shape_valid[i];
            }
        }

        //! read all data to sharemem in softmax channel
        #pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            data[i * blocksize] = in_data[input_real_index];
            input_real_index += input_stride_real[softmax_axis];
    }

        //! get maximum value in softmax channel
        dtype max_data = data[0];
        #pragma unroll
        for (int i = 1; i < axis_size; ++i) {
            dtype dt = data[i * blocksize];
            if (max_data < dt){
                max_data = dt;
            }
        }

        //! subtract then summarize
        dtype sum = 0;
        #pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            //dtype *dt = &data[i][thread_idx];
            dtype *dt = data + i * blocksize;
            *dt = expf(*dt - max_data);
            sum += *dt;
        }

        //! write back result
        #pragma unroll
        for (int i = 0; i < axis_size; ++i) {
            out_data[output_real_index] = data[i * blocksize] / sum;
            output_real_index += output_stride_real[softmax_axis];
        }
    }
}

template <>
SaberStatus SaberSoftmax<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        SoftmaxParam<NV>& param, Context<NV>& ctx) {

    //! compute size
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    CHECK_EQ(shape_in == shape_out, true) << "valid shapes must be the same";
    _outer_num = inputs[0]->count_valid(0, param.axis);
    _inner_num = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
    _axis_size = shape_in[param.axis];

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, API::get_device_id());
    size_t sharedmem_size = deviceProp.sharedMemPerBlock;
    _max_dimsize = sharedmem_size / sizeof(float) / CUDA_NUM_THREADS;

    Shape sh_tmp({1, 1, 1, _outer_num * _inner_num});
    if (_axis_size > _max_dimsize){
        //! re_alloc device memory
        _max_data.reshape(sh_tmp);
        _sum_data.reshape(sh_tmp);
    }

    //! CHECK whether the input or output tensor is with continuous buffer or not
    _is_continue_buf = outputs[0]->is_continue_mem() && inputs[0]->is_continue_mem();
    _dims = shape_in.size();
    if (!_is_continue_buf) {
        Shape sh_input_real_stride = inputs[0]->get_stride();
        Shape sh_output_real_stride = outputs[0]->get_stride();

        //! re_alloc device memory
        Shape sh({1, 1, 1, _dims});
        _valid_shape.reshape(sh);
        _input_stride.reshape(sh);
        _output_stride.reshape(sh);

        CUDA_CHECK(cudaMemcpy(_valid_shape.mutable_data(), inputs[0]->valid_shape().data(), \
                sizeof(int) * _dims, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_input_stride.mutable_data(), sh_input_real_stride.data(), \
                sizeof(int) * _dims, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_output_stride.mutable_data(), sh_output_real_stride.data(), \
                sizeof(int) * _dims, cudaMemcpyHostToDevice));
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberSoftmax<NV, AK_FLOAT>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    SoftmaxParam<NV>& param, Context<NV>& ctx) {

    //! get context
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}


template <>
SaberStatus SaberSoftmax<NV, AK_FLOAT>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    SoftmaxParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();
    //! inputs only has one tensor
    int total_threads = this->_inner_num * this->_outer_num;
    const float* data_in = (const float* )inputs[0]->data();
    float* data_out = (float*)outputs[0]->mutable_data();
    float* max_data = (float*)this->_max_data.mutable_data();
    float* sum_data = (float*)this->_sum_data.mutable_data();
    const int* valid_shape = (const int*)_valid_shape.data();  
    const int* input_stride = (const int*)_input_stride.data();
    const int* output_stride = (const int*)_output_stride.data();

    if (_is_continue_buf) {
        //! softmax kernel without roi
        if (this->_axis_size <= _max_dimsize){
            int sharemem_size = this->_axis_size * CUDA_NUM_THREADS * sizeof(float);
            sharemem_softmax_kernel<float>\
                <<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, sharemem_size, stream>>>(
                    total_threads, data_in, data_out,
                            this->_inner_num, this->_outer_num, this->_axis_size);
        } else {
            //! firstly, get maximum data
            float min_data = std::numeric_limits<float>::min();
            softmax_max_kernel<float>\
                <<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                    total_threads, data_in, max_data, min_data, \
                this->_inner_num, this->_outer_num, this->_axis_size);
            //! then, compute exp and sum data
            softmax_sub_exp_sum_kernel<float>
                    <<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                    total_threads, data_in, data_out, max_data, sum_data, \
                this->_inner_num, this->_outer_num, this->_axis_size);
            //! lastly, compute divided output
            softmax_divid_output_kernel<float>\
                <<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                    total_threads, data_out, sum_data, \
                this->_inner_num, this->_outer_num, this->_axis_size);
        }
    } else {
        //! softmax kernel with roi
        if (this->_axis_size <= _max_dimsize){
            int sharemem_size = this->_axis_size * CUDA_NUM_THREADS * sizeof(float);
            sharemem_softmax_roi_kernel<float>\
                <<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, sharemem_size, stream>>>(
                    total_threads, data_in, data_out,
                    input_stride, output_stride, valid_shape, \
                    param.axis, _axis_size, _dims);
        } else {
            //! firstly, get maximum data
            float min_data = std::numeric_limits<float>::min();
            softmax_max_roi_kernel<float>\
                <<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                    total_threads, data_in, max_data, min_data, \
                    input_stride, output_stride, valid_shape, \
                    param.axis, _axis_size, _dims);
            //! then, compute exp and sum data
            softmax_sub_exp_sum_roi_kernel<float>
                    <<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                    total_threads, data_in, data_out, max_data, sum_data, \
                    input_stride, output_stride, valid_shape, \
                    param.axis, _axis_size, _dims);
            //! lastly, compute divided output
            softmax_divid_output_roi_kernel<float>\
                <<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                    total_threads, data_out, sum_data, \
                    input_stride, output_stride, valid_shape, \
                    param.axis, _axis_size, _dims);
        }
    }

    return SaberSuccess;
}

// ============================================= int8
template <>
SaberStatus SaberSoftmax<NV, AK_INT8>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        SoftmaxParam<NV>& param, Context<NV>& ctx) {

    return SaberSuccess;
}

template <>
SaberStatus SaberSoftmax<NV, AK_INT8>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        SoftmaxParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberSoftmax<NV, AK_INT8>::dispatch(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        SoftmaxParam<NV>& param) {

    return SaberSuccess;
}

template class SaberSoftmax<NV, AK_FLOAT>;
template class SaberSoftmax<NV, AK_INT8>;
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, NV, AK_HALF);
} //namespace anakin

} //namespace anakin
