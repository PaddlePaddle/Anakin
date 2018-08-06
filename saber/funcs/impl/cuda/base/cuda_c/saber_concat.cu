#include "saber/funcs/impl/cuda/saber_concat.h"

namespace anakin{

namespace saber{

const int BLOCK_SIZE = 32;

template <typename dtype>
__global__ void concat_impl_cuda(const int nthreads, const dtype* in_data,
                            const int num_concats, const int concat_size,
                            const int top_concat_axis, const int bottom_concat_axis,
                            const int offset_concat_axis, dtype* out_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int total_concat_size = concat_size * bottom_concat_axis;
        const int concat_num = index / total_concat_size;
        const int concat_index = index % total_concat_size;
        const int top_index = concat_index +
                              (concat_num * top_concat_axis + offset_concat_axis) * concat_size;

        out_data[top_index] = in_data[index];
    }
}

template <typename dtype>
__global__ void concat_impl_2d_impl(const int inner_size, const int num_concats,
                                    const dtype* in_data, const int concat_size,
                                    const int out_concat_axis,
                                    const int offset_concat_axis, dtype* out_data) {

    int idx_inner = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_outer = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx_inner < inner_size && idx_outer < num_concats) {
        int idx_input = idx_outer * inner_size + idx_inner;
        int idx_output = (idx_outer * out_concat_axis + offset_concat_axis) * \
            concat_size + idx_inner;
        out_data[idx_output] = in_data[idx_input];
    }

}


template <>
SaberStatus SaberConcat<NV, AK_FLOAT>::dispatch(const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs, ConcatParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();

    int input_size = inputs.size();

    #if 0 //disable share memory
    //! inputs only has one tensor
    if (input_size == 1) {
        outputs[0]->set_shape(outputs[0]->valid_shape(), inputs[0]->shape(), \
                inputs[0]->offset());
        outputs[0]->share_from(*inputs[0]);
        return;
    }

    //! check whether the output is shared from input tensors
    bool share_mem = false;
    Shape offset_min = inputs[0]->offset();
    const dtype* ptr = inputs[0]->data();
    for (int i = 1; i < input_size; ++i) {
        const dtype* ptr2= inputs[i]->data();
        if (inputs[i]->offset() < offset_min) {
            offset_min = inputs[i]->offset();
        }
        share_mem = (ptr == ptr2);
        if (!share_mem){
            break;
        }
    }
    //! input tensors are sharing one tensor
    if (share_mem){
        CHECK_LE(outputs[0]->valid_size(), inputs[0]->size()) << "input shared tensors overlap";
        outputs[0]->set_shape(outputs[0]->valid_shape(), inputs[0]->shape(), offset_min);
        outputs[0]->share_from(*inputs[0]);
        return;
    }
    #endif // disable share memory

    //! get output data, valid shape and stride shape
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    int offset_concat_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[param.axis];
    bool out_cont_flag = outputs[0]->is_continue_mem();
    bool in_cont_flag = inputs[0]->is_continue_mem();
    for (int i = 1; i < input_size; ++i) {
        in_cont_flag &= inputs[i]->is_continue_mem();
    }

    //! inputs and outputs are all with continuous memory
    if (in_cont_flag && out_cont_flag){
        for (int i = 0; i < input_size; ++i) {
            Shape in_shape = inputs[i]->valid_shape();
            //std::vector<int> bottom_shape = {tmp[3], tmp[2], tmp[1], tmp[0]};
            const OpDataType* in_data = (const OpDataType*)inputs[i]->data();
            const int in_concat_axis = in_shape[param.axis];
            const int in_concat_size = in_concat_axis * _concat_input_size;
            const int nthreads = in_concat_size * _num_concats;
            float ratio = (float)in_concat_size / _num_concats;
            bool is_balance = (ratio > 0.1 && ratio < 10);
            if (is_balance){
                int block_x = BLOCK_SIZE;
                int block_y = BLOCK_SIZE;
                int grid_x = (in_concat_size + block_x - 1) / block_x;
                int grid_y = (_num_concats + block_y - 1) / block_y;
                dim3 block(block_x, block_y);
                dim3 grid(grid_x, grid_y);
                concat_impl_2d_impl<OpDataType><<<grid, block, 0, stream>>>(
                        in_concat_size, _num_concats, in_data, _concat_input_size,
                                out_concat_axis, offset_concat_axis, out_data
                );
            } else {
                // NOLINT_NEXT_LINE(whitespace/operators)
                concat_impl_cuda<OpDataType><<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, stream>>>( \
                    nthreads, in_data, _num_concats, _concat_input_size, \
                            out_concat_axis, in_concat_axis, offset_concat_axis, out_data);
            }
            offset_concat_axis += in_concat_axis;
        }
    } else { //! inputs or outputs memory is not continuous
#if 1
        Shape offset_out = outputs[0]->offset();
        Tensor<NV>  tsub;
        for (int i = 0; i < input_size; ++i) {
            Shape in_shape = inputs[i]->valid_shape();
            tsub.share_sub_buffer(*outputs[0], in_shape, offset_out);
            offset_out[param.axis] += in_shape[param.axis];
            tsub.async_copy_from(*inputs[i], stream);
        }
#endif
    }

    //outputs[0]->record_event(stream);
    return SaberSuccess;
}
#if 0
typedef Tensor<NV, AK_FLOAT, NCHW> Tensor4f_1;
typedef Tensor<NV, AK_FLOAT, NHWC> Tensor4f_2;
typedef Tensor<NV, AK_FLOAT, HW> Tensor2f;
typedef Tensor<NV, AK_INT8, NCHW> Tensor4c_1;
typedef Tensor<NV, AK_INT8, NHWC> Tensor4c_2;
typedef Tensor<NV, AK_INT8, HW> Tensor2c;
template SaberStatus SaberConcat<AK_FLOAT, NCHW>::dispatch(const std::vector<Tensor4f_1 *> inputs, std::vector<Tensor4f_1 *> outputs,
                                              ConcatParam<Tensor4f_1> &param);
template SaberStatus SaberConcat<AK_FLOAT, NHWC>::dispatch(const std::vector<Tensor4f_2 *> inputs, std::vector<Tensor4f_2 *> outputs,
                                              ConcatParam<Tensor4f_2> &param);
template SaberStatus SaberConcat<AK_FLOAT, HW>::dispatch(const std::vector<Tensor2f *> inputs, std::vector<Tensor2f *> outputs,
                                              ConcatParam<Tensor2f> &param);
template SaberStatus SaberConcat<AK_INT8, NCHW>::dispatch(const std::vector<Tensor4c_1 *> inputs, std::vector<Tensor4c_1 *> outputs,
                                              ConcatParam<Tensor4c_1> &param);
template SaberStatus SaberConcat<AK_INT8, NHWC>::dispatch(const std::vector<Tensor4c_2 *> inputs, std::vector<Tensor4c_2 *> outputs,
                                                   ConcatParam<Tensor4c_2> &param);
template SaberStatus SaberConcat<AK_INT8, HW>::dispatch(const std::vector<Tensor2c *> inputs, std::vector<Tensor2c *> outputs,
                                                   ConcatParam<Tensor2c> &param);
#endif
} //namespace anakin

} //namespace anakin
