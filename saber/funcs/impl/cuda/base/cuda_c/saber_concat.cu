#include "saber/funcs/impl/cuda/saber_concat.h"
#include "saber/funcs/impl/cuda/reorder.h"
#include "saber/funcs/calibrate.h"

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
SaberStatus SaberConcat<NV, AK_FLOAT>::create(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConcatParam<NV>& param,
        Context<NV>& ctx) {

    _num_concats = inputs[0]->count_valid(0, param.axis);
    _concat_input_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
    return SaberSuccess;
}

template <>
SaberStatus SaberConcat<NV, AK_FLOAT>::init(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConcatParam<NV>& param,
        Context<NV> &ctx) {
    // get context
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConcat<NV, AK_FLOAT>::dispatch(const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs, ConcatParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();

    int input_size = inputs.size();
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
            if (is_balance) {
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
        Shape offset_out = outputs[0]->offset();
        Tensor<NV> tsub;
        for (int i = 0; i < input_size; ++i) {
            Shape in_shape = inputs[i]->valid_shape();
            tsub.share_sub_buffer(*outputs[0], in_shape, offset_out);
            offset_out[param.axis] += in_shape[param.axis];
            tsub.async_copy_from(*inputs[i], stream);
        }
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberConcat<NV, AK_INT8>::create(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConcatParam<NV>& param,
        Context<NV>& ctx) {

    _num_concats = inputs[0]->count_valid(0, param.axis);
    _concat_input_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
    _input_v.resize(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {

        if (inputs[i]->get_dtype() == AK_FLOAT) {
            _input_v[i].re_alloc(inputs[i]->valid_shape(), AK_INT8);
        } else if (inputs[i]->get_dtype() == AK_INT8 && inputs[i]->get_layout() == Layout_NCHW_C4) {
            Shape new_shape = Shape({inputs[i]->num(), inputs[i]->channel(),
                                     inputs[i]->height(), inputs[i]->width()}, Layout_NCHW);
            _input_v[i].re_alloc(new_shape, AK_INT8);
        } else if (inputs[i]->get_dtype() == AK_INT8 && inputs[i]->get_layout() == Layout_NCHW) {
            // good, nothing to do
        } else {
            LOG(FATAL) << "Not support this situation, pls contact the r&d.";
        }
    }

    if (outputs[0]->get_dtype() == AK_FLOAT) {
        _output.re_alloc(outputs[0]->valid_shape(), AK_INT8);
        _output.set_scale(outputs[0]->get_scale());
    } else if (outputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_layout() == Layout_NCHW_C4) {
        Shape new_shape = outputs[0]->valid_shape();
        new_shape.set_layout(Layout_NCHW);
        _output.re_alloc(new_shape, AK_INT8);
        _output.set_scale(outputs[0]->get_scale());
    } else if (outputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_layout() == Layout_NCHW) {
        // good, nothing to do.
    } else {
        LOG(FATAL) << "Not support this situation, pls contact the r&d.";
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberConcat<NV, AK_INT8>::init(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConcatParam<NV>& param,
        Context<NV> &ctx) {
    // get context
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConcat<NV, AK_INT8>::dispatch(const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs, ConcatParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();
    int input_size = inputs.size();
    //! get output data, valid shape and stride shape
    char* out_data = nullptr;

    if (outputs[0]->get_dtype() == AK_FLOAT) {
        out_data = (char*)_output.mutable_data();
    } else if (outputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_layout() == Layout_NCHW_C4) {
        out_data = (char*)_output.mutable_data();
    } else if (outputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_layout() == Layout_NCHW) {
        out_data = (char*)outputs[0]->mutable_data();
    } else {
        LOG(FATAL) << "Not support this situation, pls contact the r&d.";
    }

    int offset_concat_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[param.axis];

    //! inputs and outputs are all with continuous memory
    for (int i = 0; i < input_size; ++i) {
        Shape in_shape = inputs[i]->valid_shape();
        //std::vector<int> bottom_shape = {tmp[3], tmp[2], tmp[1], tmp[0]};
        const char* in_data = nullptr;
        if (inputs[i]->get_dtype() == AK_FLOAT) {
            flatten_calibrate<NV, char, float> (_input_v[i], *inputs[i], *_ctx);
            in_data = (char*)_input_v[i].mutable_data();
        } else if (inputs[i]->get_dtype() == AK_INT8 && inputs[i]->get_layout() == Layout_NCHW_C4) {
            convert_nchwc4_to_nchw<NV>(_input_v[i], *inputs[i], *_ctx);
            in_data = (char*)_input_v[i].mutable_data();
        } else if (inputs[i]->get_dtype() == AK_INT8 && inputs[i]->get_layout() == Layout_NCHW) {
            in_data = (char*)inputs[i]->mutable_data();
        } else {
            LOG(FATAL) << "Not support this situation, pls contact the r&d.";
        }
        const int in_concat_axis = in_shape[param.axis];
        const int in_concat_size = in_concat_axis * _concat_input_size;
        const int nthreads = in_concat_size * _num_concats;
        float ratio = (float)in_concat_size / _num_concats;
        bool is_balance = (ratio > 0.1 && ratio < 10);
        if (is_balance) {
            int block_x = BLOCK_SIZE;
            int block_y = BLOCK_SIZE;
            int grid_x = (in_concat_size + block_x - 1) / block_x;
            int grid_y = (_num_concats + block_y - 1) / block_y;
            dim3 block(block_x, block_y);
            dim3 grid(grid_x, grid_y);
            concat_impl_2d_impl<char><<<grid, block, 0, stream>>>(
                    in_concat_size, _num_concats, in_data, _concat_input_size,
                    out_concat_axis, offset_concat_axis, out_data);
        } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
            concat_impl_cuda<char><<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, stream>>>(
                    nthreads, in_data, _num_concats, _concat_input_size,
                    out_concat_axis, in_concat_axis, offset_concat_axis, out_data);
        }
        offset_concat_axis += in_concat_axis;
    }
    if (outputs[0]->get_dtype() == AK_FLOAT) {
        flatten_calibrate<NV, float, char>(*outputs[0], _output, *_ctx);
    } else if (outputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_layout() == Layout_NCHW_C4) {
        convert_nchw_to_nchwc4<NV>(*outputs[0], _output, *_ctx);
    } else if (outputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_layout() == Layout_NCHW) {
        // good, nothing to be done;
    } else {
        LOG(FATAL) << "Not support this situation, pls contact the r&d.";
    }
    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, NV, AK_HALF);

} //namespace anakin

} //namespace anakin
