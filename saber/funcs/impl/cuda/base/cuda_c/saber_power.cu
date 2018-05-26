#include "saber/funcs/impl/cuda/saber_power.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_power_fwd(Dtype * out_data, \
                    const int count, const float scale,\
                    const float shift, const float power,\
                    const Dtype* in_data)
{
    CUDA_KERNEL_LOOP(tid, count){
        out_data[tid] = pow(in_data[tid] * scale + shift, power);
    }
}

template <typename Dtype>
__global__ void ker_scale_fwd(Dtype * out_data, \
                    const int count, const float scale,\
                    const float shift,\
                    const Dtype* in_data)
{
    CUDA_KERNEL_LOOP(tid, count){
        out_data[tid] = in_data[tid] * scale + shift;
    }
}

template <typename Dtype>
__global__ void ker_power_fwd(Dtype * out_data, \
                    const int count, const float scale,\
                    const float shift, const float power,\
                    const int* out_shape,
                    const int* out_stride,
                    const int* in_stride,
                    const int num_axis,
                    const Dtype* in_data)
{
    CUDA_KERNEL_LOOP(tid, count){
        int in_offset = 0;
        int out_offset = 0;
        int valid_stride = 1;
        for (int i = num_axis - 1; i >= 0; --i) {
             int id = (tid / valid_stride) % out_shape[i];
             in_offset += id * in_stride[i];
             out_offset += id * out_stride[i];
             valid_stride *= out_shape[i];
        }
        out_data[out_offset] = pow(in_data[in_offset] * scale + shift, power);
    }
}

template <typename Dtype>
__global__ void ker_scale_fwd(Dtype * out_data, \
                    const int count, const float scale,\
                    const float shift,\
                    const int* out_shape,
                    const int* out_stride,
                    const int* in_stride,
                    const int num_axis,
                    const Dtype* in_data)
{
    CUDA_KERNEL_LOOP(tid, count){
        int in_offset = 0;
        int out_offset = 0;
        int valid_stride = 1;
        for (int i = num_axis - 1; i >= 0; --i) {
             int id = (tid / valid_stride) % out_shape[i];
             in_offset += id * in_stride[i];
             out_offset += id * out_stride[i];
             valid_stride *= out_shape[i];
        }
        //printf("%d, %d, %d\n", tid, in_offset, out_offset);
        out_data[out_offset] = in_data[in_offset] * scale + shift;
        //printf("out_offset:%d, %f\n", out_offset, out_data[out_offset]);
    }
}

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
SaberStatus SaberPower<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    PowerParam<OpTensor>& param) {

    const InDataType* in_data = inputs[0]->data();
    OutDataType* out_data = outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx.get_compute_stream();
    int count = outputs[0]->valid_size();
    const float scale = param.scale;
    const float shift = param.shift;
    const float power = param.power;
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        if (power == 1) {
            ker_scale_fwd<OpDataType>\
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                    out_data, count, scale, shift, in_data);
        } else {
            ker_power_fwd<OpDataType>\
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                    out_data, count, scale, shift, power, in_data);
        }
    } else {
        const int* i_stride = _in_steps.data();
        const int* o_stride = _out_steps.data();
        const int* valid_shape = _out_valid_shape.data();
        if (power == 1) {
            ker_scale_fwd<OpDataType>\
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                    out_data, count, scale, shift, valid_shape, 
                    o_stride, i_stride, outputs[0]->dims(), in_data);
        } else {
            ker_power_fwd<OpDataType>\
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                    out_data, count, scale, shift, power, valid_shape,
                    o_stride, i_stride, outputs[0]->dims(), in_data);
        }
    }

    return SaberSuccess;
}

}
}
