#include "saber/funcs/impl/cuda/saber_ctc_align.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_ctc_align_fwd(Dtype * out_data, \
                    int* out_offset,
                    const Dtype* in_data,
                    const int* in_offset,
                    const int seq_num,
                    const int blank,
                    const bool merge_repeated,
                    const int num_threads)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        int index = 0;
        for (int seq_id = 0; seq_id < seq_num; seq_id++) {
            Dtype prev_token = -1;
            out_offset[seq_id] = index;
            for (int i = in_offset[seq_id]; i < in_offset[seq_id + 1]; i++) {
                if (in_data[i] != blank && !(merge_repeated && in_data[i] == prev_token)) {
                    out_data[index++] = in_data[i];
                    prev_token = in_data[i];
                }
            }
        }
        out_offset[seq_num] = index;
    }
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberCtcAlign<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(const std::vector<DataTensor_in *>& inputs,
    std::vector<DataTensor_out *>& outputs,
    CtcAlignParam<OpTensor>& param) {

    const InDataType* in_data = inputs[0]->data();
    OutDataType* out_data = outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    int out_n = outputs[0]->num();
    int* in_offset = _in_offset.mutable_data();
    int* out_offset = _out_offset.mutable_data();
    int seq_num = (inputs[0]->get_seq_offset()).size() - 1;
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        cudaMemcpyAsync(in_offset, &(inputs[0]->get_seq_offset())[0], sizeof(int) * (seq_num + 1), cudaMemcpyHostToDevice, cuda_stream);
           ker_ctc_align_fwd<InDataType>\
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                    out_data, out_offset, in_data, \
                    in_offset, seq_num, param.blank, param.merge_repeated,
                    1);

        std::vector<int> seq_offset;
        seq_offset.resize((inputs[0]->get_seq_offset()).size());
        cudaMemcpyAsync(&seq_offset[0], out_offset, sizeof(int) * (seq_num + 1), cudaMemcpyDeviceToHost, cuda_stream);
        outputs[0]->set_seq_offset(seq_offset);
    }

    return SaberSuccess;
}

}
}
