
#include "core/common.h"
#include "saber/funcs/impl/cuda/saber_sequence_pool_concat.h"
#include "saber/saber_funcs_param.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberSequencePoolConcat<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequencePoolConcatParam<NV>& param, Context<NV>& ctx) {
    if (inputs[0]->get_seq_offset().size() > 0 && inputs[0]->get_seq_offset()[0].size() > 0) {
        auto offset = inputs[0]->get_seq_offset()[0];
        auto stream = _ctx->get_compute_stream();

        _offset_buffer.re_alloc(offset.size() * sizeof(float));
        cudaMemcpyAsync(_offset_buffer.get_data_mutable(), offset.data(),
            offset.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberSequencePoolConcat<NV, AK_FLOAT>::init(
    const std::vector<Tensor<NV>*>& inputs,
    std::vector<Tensor<NV>*>& outputs,
    SequencePoolConcatParam<NV>& param, Context<NV>& ctx) {

    _ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

__global__
void sequence_pool_sum_concat(const float* input_data,
        float* output_data, const int* offset, int n_total, int xdim) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gid = bid * blockDim.x + tid;
    int n_idx = gid / xdim;
    int feature_num;
    int x_idx = gid % xdim;
    if (n_idx < n_total) {
        feature_num = offset[n_idx + 1] - offset[n_idx];
        float* out_data = output_data + n_idx * xdim;
        const float* in_data = input_data + offset[n_idx] * xdim;
        float res = 0.f;
        for (int i = 0; i < feature_num; ++i) {
            res += in_data[x_idx];
            in_data += xdim;
        }
//        printf("gid = %d, feature_num = %d, n_idx = %d, xdim = %d feature_num = %d idx = %d\n", gid, feature_num, n_idx, xdim, feature_num, offset[n_idx] * xdim);
        out_data[x_idx] = res;
    }
}

template <>
SaberStatus SaberSequencePoolConcat<NV, AK_FLOAT>::dispatch(
    const std::vector<Tensor<NV>*>& inputs,
    std::vector<Tensor<NV>*>& outputs,
    SequencePoolConcatParam<NV>& param) {

    CHECK_GE(inputs[0]->get_seq_offset().size(), 1);
    auto offset = inputs[0]->get_seq_offset()[0];
    CHECK_GE(offset.size(), 1);
    auto stream = _ctx->get_compute_stream();

    int slot_num = param.slot_num;
    int batch = (offset.size() - 1) / slot_num;
    int xdim = outputs[0]->valid_size();
    CHECK_EQ((xdim % slot_num), 0) << "some data is wrong!!!" << xdim << " " << slot_num;
    CHECK_GE(batch, 1);
    xdim /= slot_num;
    xdim /= batch;
    int count = slot_num * batch * xdim;

    const float* in_data = (const float*)inputs[0]->data();
    float* out_data = (float*)outputs[0]->mutable_data();
    const int* offset_data = (const int*)_offset_buffer.get_data();
    switch (param.sequence_pool_param.sequence_pool_type) {
        case Sequence_pool_sum:
            sequence_pool_sum_concat<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>> (
                    in_data, out_data, offset_data, slot_num * batch, xdim);
            break;
        default:
            LOG(FATAL) << "not implemented yet!!!";
            break;
    }
    //cudaDeviceSynchronize();

    return SaberSuccess;
}

template class SaberSequencePoolConcat<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSequencePoolConcat, SequencePoolConcatParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequencePoolConcat, SequencePoolConcatParam, NV, AK_INT8);
}
} // namespace anakin
