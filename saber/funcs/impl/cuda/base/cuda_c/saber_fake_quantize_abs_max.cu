#include "saber/funcs/impl/cuda/saber_fake_quantize_abs_max.h"
#include "cuda_fp16.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberFakeQuantizeAbsMax<NV, AK_FLOAT>::\
    create(const std::vector<Tensor<NV> *>& inputs,
           std::vector<Tensor<NV> *>& outputs,
           FakeQuantizeAbsMaxParam<NV>& param, Context<NV>& ctx) {
    if (&ctx != this->_ctx) {
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
        this->_ctx = &ctx;
        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();
        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));
    }

    int input_num = inputs[0]->num();
    int input_channel = inputs[0]->channel();
    int input_height = inputs[0]->height();
    int input_width = inputs[0]->width();


    Shape in_stride = inputs[0]->get_stride();
    Shape max_abs_stride = std::vector<int>{1, 1, 1, 1};

    int dim_a[] = {input_num, input_channel,
                   input_height, input_width};
    int dim_b[] = {1, 1, 1, 1};

    cudnn::setTensorNdDesc<float >(&_input_descs,
                                        inputs[0]->dims(), dim_a, &in_stride[0]);

    cudnn::setTensorNdDesc<float>(&_output_descs,
                                        _max_abs.dims(), dim_b, &max_abs_stride[0]);

    cudnn::setReduceTensorDesc<OpDataType >(&_reduce_tensor_descs,
                                             CUDNN_REDUCE_TENSOR_AMAX,
                                             CUDNN_PROPAGATE_NAN,
                                             CUDNN_REDUCE_TENSOR_NO_INDICES,
                                             CUDNN_64BIT_INDICES);

    // Get fastest implement of cudnn
    // set up algo and workspace size
   size_t workspace_size = 0;

    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            _handle, _reduce_tensor_descs,  _input_descs, _output_descs, &workspace_size));

    if (workspace_size > _workspaceSizeInBytes) {
        _workspaceSizeInBytes = workspace_size;
        if (_workspace != NULL) {
            cudaFree(_workspace);
        }
        cudaMalloc(&_workspace, _workspaceSizeInBytes);
    }

    size_t indices_size = 0;
    CUDNN_CHECK(cudnnGetReductionIndicesSize(_handle, _reduce_tensor_descs,
        _input_descs, _output_descs, &indices_size));
    if (indices_size > _indices_size) {
        _indices_size = indices_size;
        if (_indices != NULL) {
            cudaFree(_indices);
        }
        cudaMalloc(&_indices, _indices_size);
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberFakeQuantizeAbsMax<NV, AK_FLOAT>::\
    init(const std::vector<Tensor<NV> *>& inputs,
           std::vector<Tensor<NV> *>& outputs,
           FakeQuantizeAbsMaxParam<NV>& param, Context<NV>& ctx) {
    _workspaceSizeInBytes = 0;
    _workspace = NULL;
    _indices = NULL;
    _indices_size = 0;

    this->_ctx = &ctx;
    // ---- get cuda resources ----
    cudaStream_t cuda_stream;
    cuda_stream = ctx.get_compute_stream();
    CUDNN_CHECK(cudnnCreate(&_handle));
    CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

    int in_channels = inputs[0]->channel();
    // ---- create cudnn Descs ----
    cudnn::createReduceTensorDesc<OpDataType>(&_reduce_tensor_descs);
    cudnn::createTensorDesc<OpDataType>(&_input_descs);
    cudnn::createTensorDesc<OpDataType>(&_output_descs);
    Shape max_abs_shape = std::vector<int>{1, 1, 1, 1};
    _max_abs.reshape(max_abs_shape);

    return create(inputs, outputs, param, ctx);
}


template <typename Dtype, typename Ttype>
__global__ void ker_fake_quantize_max_abs_fwd(Ttype * out_data, \
                    const Dtype* in_data,
                    const Dtype scale,
                    const int count)
{
    CUDA_KERNEL_LOOP(tid, count){
        out_data[tid] = round(in_data[tid] * scale);
        //printf("%d, %d\n", tid, (int)out_data[tid]);
    }
}


template <DataType OpDtype>
SaberStatus SaberFakeQuantizeAbsMax<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    FakeQuantizeAbsMaxParam<NV>& param) {
    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    OpDataType* max_abs_data = (OpDataType*) _max_abs.mutable_data();

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    float alpha = 1.0f;
    float beta = 0.f;
    OpDataType cpu_max_abs_data;

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        cudnnReduceTensor(_handle,
                         _reduce_tensor_descs,
                         _indices,
                         _indices_size,
                         _workspace,
                         _workspaceSizeInBytes, 
                         &alpha, 
                         _input_descs,
                         in_data,
                         &beta,
                         _output_descs,
                         max_abs_data);
        cudaMemcpyAsync((void*)&cpu_max_abs_data, (void*)max_abs_data, sizeof(OpDataType) * 1, cudaMemcpyDeviceToHost, cuda_stream);
        OpDataType scale = ((1 << (param.bit_length - 1)) - 1) / cpu_max_abs_data;
        auto out_data = outputs[0]->mutable_data();
        //LOG(INFO) <<"gpu max_data" << cpu_max_abs_data;
        if (param.bit_length == 8) {
            ker_fake_quantize_max_abs_fwd<OpDataType, char>\
                     <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                     (char*)out_data, in_data, \
                     scale, count);
        } else if (param.bit_length == 16) {
            ker_fake_quantize_max_abs_fwd<OpDataType, int16_t>\
                     <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                     (int16_t*)out_data, in_data, \
                     scale, count);
        } else {
            LOG(FATAL) << "other bit length has not been supported";
        }
    }

    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberFakeQuantizeAbsMax, FakeQuantizeAbsMaxParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberFakeQuantizeAbsMax, FakeQuantizeAbsMaxParam, NV, AK_INT8);
}
}
