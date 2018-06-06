#include "saber/funcs/impl/cuda/saber_prelu.h"

namespace anakin{

namespace saber{

template <typename dtype>
__global__ void prelu_shared_kernel(int n, const dtype* slope, const dtype* src, dtype* dst) {
    CUDA_KERNEL_LOOP(idx, n) {
        dst[idx] = src[idx] > 0 ? src[idx] : src[idx] * slope[0];
    }
}

template <typename dtype>
__global__ void prelu_kernel(int n, int channels, int inner_size, \
    const dtype* slope, const dtype* src, dtype* dst) {

    CUDA_KERNEL_LOOP(idx, n) {
        int c = (idx / inner_size) % channels;
        dst[idx] = src[idx] > 0 ? src[idx] : src[idx] * slope[c];
    }
}

template <typename dtype>
__global__ void prelu_shared_roi_kernel(int n, int dims, \
        const int* input_stride_real, const int* output_stride_real, const int* shape_valid, \
        const dtype* slope, const dtype* src, dtype* dst) {

    CUDA_KERNEL_LOOP(idx, n) {
        int index = idx;
        //! compute real data index
        int input_real_index = 0;
        int output_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            int x = index % shape_valid[i];
            input_real_index += x * input_stride_real[i];
            output_real_index += x * output_stride_real[i];
            index = index / shape_valid[i];
        }
        dst[output_real_index] = src[input_real_index] > 0 ? src[input_real_index] : \
            src[input_real_index] * slope[0];
    }
}

template <typename dtype>
__global__ void prelu_roi_kernel(int n, int channels, int inner_size, int dims, \
        const int* input_stride_real, const int* output_stride_real, const int* shape_valid, \
        const dtype* slope, const dtype* src, dtype* dst) {

    CUDA_KERNEL_LOOP(idx, n) {
        int index = idx;
        //! compute real data index
        int input_real_index = 0;
        int output_real_index = 0;
        for (int i = dims - 1; i >= 0; i--) {
            int x = index % shape_valid[i];
            input_real_index += x * input_stride_real[i];
            output_real_index += x * output_stride_real[i];
            index = index / shape_valid[i];
        }
        int c = (idx / inner_size) % channels;
        dst[output_real_index] = src[input_real_index] > 0 ? src[input_real_index] : \
            src[input_real_index] * slope[c];
    }
}


template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
SaberStatus SaberPrelu<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    PreluParam<OpTensor>& param) {
    cudaStream_t stream = this->_ctx.get_compute_stream();

    const InDataType* src = inputs[0]->data();
    OutDataType* dst = outputs[0]->mutable_data();
    const int* valid_shape = _valid_shape.data();
    const int* input_stride = _input_stride.data();
    const int* output_stride = _output_stride.data();

    if (_is_continue_buf) {
        if (param.channel_shared) {
            prelu_shared_kernel<OpDataType><<<CUDA_GET_BLOCKS(_size), CUDA_NUM_THREADS, 0, stream>>>\
                (_size, param.slope->data(), src, dst);
        } else {
            prelu_kernel<OpDataType><<<CUDA_GET_BLOCKS(_size), CUDA_NUM_THREADS, 0, stream>>>\
                (_size, _channels, _inner_size, param.slope->data(), src, dst);
        }
    } else {
        if (param.channel_shared) {
            prelu_shared_roi_kernel<OpDataType>\
                <<<CUDA_GET_BLOCKS(_size), CUDA_NUM_THREADS, 0, stream>>>\
                (_size, _dims, input_stride, output_stride, valid_shape, \
                    param.slope->data(), src, dst);
        } else {
            prelu_roi_kernel<OpDataType>\
                <<<CUDA_GET_BLOCKS(_size), CUDA_NUM_THREADS, 0, stream>>>\
                (_size, _channels, _inner_size, _dims, input_stride, \
                output_stride, valid_shape, param.slope->data(), src, dst);
        }
    }

    return SaberSuccess;
}

} //namespace anakin

} //namespace anakin
