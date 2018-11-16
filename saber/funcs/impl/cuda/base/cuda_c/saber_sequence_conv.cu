#include "saber/funcs/impl/cuda/saber_sequence_conv.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/tensor_op.h"
#include "sass_funcs.h"
namespace anakin {
namespace saber {

static void gemm(const bool TransA, const bool TransB, int m, int n, int k, const float alpha,
                 const float* a, const float* b, const float beta, float* c, Context<NV>* ctx) {
    saber_find_fast_sass_gemm(TransA, TransB, m, n, k)\
                                (m, n ,k, alpha, a, beta, b, c, ctx->get_compute_stream());
}
template<typename Dtype>
__global__ void  im2col_2d_ocf_kernel(const Dtype* in, int stride, int pad_up, int pad_down, int kernel_size,
                          Dtype* out, int seq_length, int hidden_size){
    int hidden_kernel_size = hidden_size * kernel_size;
    int in_size = seq_length * hidden_size;
    int tid = threadIdx.x;
    int out_h = blockIdx.x;
    int in_index = (blockIdx.x - pad_up) * hidden_size * stride;
    while (tid < hidden_kernel_size){
        int out_index = out_h * hidden_kernel_size + tid;
        if (in_index + tid < 0 || in_index + tid >= in_size){
            out[out_index] = Dtype(0);
        }
        else {
            out[out_index] = in[in_index + tid];
        }

        tid += CUDA_NUM_THREADS;
    }
}

template<typename Dtype>
static void im2col_2d_ocf(const Dtype* in, int stride, int pad_up, int pad_down, int kernel_size,
                            Dtype* out, int seq_length, int hidden_size) {
    int blockdim = seq_length;
    im2col_2d_ocf_kernel<<<blockdim, CUDA_NUM_THREADS>>>(in, stride, pad_up, pad_down, kernel_size,
							out, seq_length, hidden_size);
}
template <>
SaberStatus SaberSequenceConv<NV, AK_FLOAT>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SequenceConvParam<NV>& param) {

    DataTensor_in* in_data = inputs[0];
    DataTensor_out* out_data = outputs[0];

    std::vector<std::vector<int>> voffset = in_data->get_seq_offset();
    CHECK_GT(voffset.size(), 0) << "seq_offset size should greater than 1";
    std::vector<int> offset = voffset[0];
    int word_num = offset[offset.size() - 1];
	Shape sh_im({1, 1, word_num, param.filter_tensor->height()});
    _temp_im2col_tensor.re_alloc(sh_im, AK_FLOAT);

	const float* in = (const float*)in_data->data();
	float* out = (float*)out_data->mutable_data();
	float* im2col = (float*)_temp_im2col_tensor.mutable_data();

    for (int i = 0; i < offset.size() - 1; ++i) {
        int start = offset[i];
        int seq_length = offset[i + 1] - offset[i];
        im2col_2d_ocf(in + _hidden_size * start, param.context_stride, _up_pad, _down_pad,
                      param.context_length, im2col + _hidden_kernel_size * start, seq_length,
                      _hidden_size);
    }

    gemm(false, false, word_num, _feature_size, _hidden_kernel_size, 1.f, (const float*)im2col,
         (const float*)param.filter_tensor->data(), 0.f, out, this->_ctx);

    out_data->set_seq_offset(voffset);

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberSequenceConv, SequenceConvParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequenceConv, SequenceConvParam, NV, AK_INT8);

}
}
