#include "saber/funcs/impl/cuda/saber_slice.h"

namespace anakin{

namespace saber{

template <typename dtype>
__global__ void slice_impl_cuda(const int nthreads, const dtype* in_data,
                                const int num_slices, const int slice_size,
                                const int in_slice_axis_size, const int out_slice_axis_size,
                                const int offset_slice_axis, dtype* out_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int total_slice_size = slice_size * out_slice_axis_size;
        const int slice_num = index / total_slice_size;
        const int slice_index = index % total_slice_size;
        const int in_index = slice_index +
                                 (slice_num * in_slice_axis_size + offset_slice_axis) * slice_size;
        out_data[index] = in_data[in_index];
    }
}


template <DataType OpDtype>
SaberStatus SaberSlice<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    SliceParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();
    //! inputs only has one tensor
    Shape shape_in = inputs[0]->valid_shape();

    int output_size = outputs.size();

    if (output_size == 1) {
        outputs[0]->share_from(*inputs[0]);
        return SaberSuccess;
    }

    int offset_slice_axis = 0;
    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    const int in_slice_axis_size = shape_in[param.axis];
    for (int i = 0; i < output_size; ++i) {
        OpDataType* out_data = (OpDataType*)outputs[i]->mutable_data();
        const int out_slice_axis_size = outputs[i]->valid_shape()[param.axis];
        const int out_slice_size = out_slice_axis_size * _slice_size;
        const int nthreads = out_slice_size * _slice_num;
        slice_impl_cuda<OpDataType><<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, stream>>>(
                nthreads, in_data, _slice_num, _slice_size,
                        in_slice_axis_size, out_slice_axis_size, offset_slice_axis, out_data);
        offset_slice_axis += out_slice_axis_size;
    }
    return SaberSuccess;

}
DEFINE_OP_TEMPLATE(SaberSlice, SliceParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSlice, SliceParam, NV, AK_INT8);
} //namespace anakin

} //namespace anakin
