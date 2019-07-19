
#include "saber/funcs/impl/cuda/vender_reduce.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/funcs/debug.h"

namespace anakin {
namespace saber {

template <>
SaberStatus VenderReduce<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ReduceParam<NV>& param, Context<NV>& ctx) {

    if (&ctx != this->_ctx) {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
        this->_ctx = &ctx;
        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, ctx.get_compute_stream()));
    }

    int input_num = inputs[0]->num();
    int input_channel = inputs[0]->channel();
    int input_height = inputs[0]->height();
    int input_width = inputs[0]->width();
    int output_num = outputs[0]->num();
    int output_channel = outputs[0]->channel();
    int output_height = outputs[0]->height();
    int output_width = outputs[0]->width();

    Shape in_stride = inputs[0]->get_stride();
    Shape out_stride = outputs[0]->get_stride();

    int dim_a[] = {input_num, input_channel,
                   input_height, input_width};
    int dim_b[] = {output_num, output_channel,
                   output_height, output_width};

    cudnn::setTensorNdDesc<float >(&_input_descs,
            inputs[0]->dims(), dim_a, &in_stride[0]);

    cudnn::setTensorNdDesc<float>(&_output_descs,
            outputs[0]->dims(), dim_b, &out_stride[0]);

    // todo add the parameters.

    cudnnReduceTensorOp_t _reduce_tensor_op = CUDNN_REDUCE_TENSOR_MIN;
    switch (param.reduce_type) {
        case Reduce_min:
            _reduce_tensor_op = CUDNN_REDUCE_TENSOR_MIN;
            break;
        case Reduce_max:
            _reduce_tensor_op = CUDNN_REDUCE_TENSOR_MAX;
            break;
        case Reduce_sum:
            _reduce_tensor_op = CUDNN_REDUCE_TENSOR_ADD;
            break;
        case Reduce_avg:
            _reduce_tensor_op = CUDNN_REDUCE_TENSOR_AVG;
            break;
        case Reduce_prod:
            _reduce_tensor_op = CUDNN_REDUCE_TENSOR_MUL;
            break;
        default:
            LOG(FATAL) << "param reduce_type is unknown!!!!";
            break;
    }

    cudnnDataType_t _reduce_tensor_comp_type = CUDNN_DATA_FLOAT;
    cudnnNanPropagation_t _reduce_tensor_nan_opt = CUDNN_NOT_PROPAGATE_NAN;
    cudnnReduceTensorIndices_t _reduce_tensor_indices = CUDNN_REDUCE_TENSOR_NO_INDICES;
    cudnnIndicesType_t _reduce_tensor_indices_type = CUDNN_32BIT_INDICES;

    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(_reduce_descs,
            _reduce_tensor_op,
            _reduce_tensor_comp_type,
            _reduce_tensor_nan_opt,
            _reduce_tensor_indices,
            _reduce_tensor_indices_type));

    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            _handle, _reduce_descs, _input_descs, _output_descs,
            &_workspace_fwd_sizes));

    if (_workspace != NULL) {
        cudaFree(_workspace);
    }
    cudaMalloc(&_workspace, _workspace_fwd_sizes);

    return SaberSuccess;
}

template <>
SaberStatus VenderReduce<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ReduceParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, ctx.get_compute_stream()));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&_input_descs));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&_output_descs));
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&_reduce_descs));

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderReduce<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ReduceParam<NV>& param) {

    const void * in_data = inputs[0]->data();
    void* out_data = outputs[0]->mutable_data();
    float alpha = param.coeff;// should be 1 for default impl.
    float beta = 0.f;
    CUDNN_CHECK(cudnnReduceTensor(_handle, _reduce_descs,
            nullptr, 0,
            _workspace, _workspace_fwd_sizes,
            &alpha, _input_descs, in_data,
            &beta, _output_descs, out_data));
    return SaberSuccess;
}

template class VenderReduce<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderReduce, ReduceParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderReduce, ReduceParam, NV, AK_INT8);

} // namespace saber.
} // namespace anakin.