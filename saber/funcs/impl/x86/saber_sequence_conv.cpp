#include "saber/funcs/impl/x86/saber_sequence_conv.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/tensor_op.h"
#include "mkl_cblas.h"
namespace anakin {
namespace saber {

static void gemm(const bool TransA, const bool TransB, int m, int n, int k, const float alpha,
                 const float* a, const float* b, const float beta, float* c) {

    int lda = (!TransA) ? k : m;
    int ldb = (!TransB) ? n : k;
    CBLAS_TRANSPOSE cuTransA =
        (!TransA) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cuTransB =
        (!TransB) ? CblasNoTrans : CblasTrans;
    cblas_sgemm(CblasRowMajor, cuTransA, cuTransB, m, n, k, alpha, a, k, b, n, beta, c, n);
};

template <typename Dtype>
static void im2col_2d_ocf(const Dtype* in, int stride, int pad_up, int pad_down, int kernel_size,
                          Dtype* out, int seq_length, int hidden_size) {
    for (int out_row = 0; out_row < seq_length; ++out_row) {
        for (int col = 0; col < kernel_size; ++col) {
            int index = out_row + col - pad_up;
            int out_index = (out_row * kernel_size + col) * hidden_size;

            for (int hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
                if (index < 0 || index >= seq_length) {
                    out[out_index + hidden_index] = 0;
                } else {
                    out[out_index + hidden_index] = in[index * hidden_size + hidden_index];
                }
            }
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberSequenceConv<X86, OpDtype>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SequenceConvParam<X86>& param) {
    DataTensor_in* in_data = inputs[0];
    DataTensor_out* out_data = outputs[0];
    std::vector<int> offset = in_data->get_seq_offset()[0];

    int word_num = offset[offset.size() - 1];
    Shape sh_im({1, 1, word_num, param.filter_tensor->height()});
    _temp_im2col_tensor.re_alloc(sh_im, AK_FLOAT);

    for (int i = 0; i < offset.size() - 1; ++i) {
        int start = offset[i];
        int seq_length = offset[i + 1] - offset[i];
        im2col_2d_ocf<float>((const float *)in_data->data() + _hidden_size * start, param.context_stride, _up_pad, _down_pad,
                      param.context_length, (float*)_temp_im2col_tensor.mutable_data() + _hidden_kernel_size * start, seq_length,
                      _hidden_size);
    }

    gemm(false, false, word_num, _feature_size, _hidden_kernel_size, 1.f, static_cast<const float*>(_temp_im2col_tensor.data()),
         static_cast<const float*>(param.filter_tensor->data()), 0.f, static_cast<float*>(out_data->mutable_data()));
    std::vector<std::vector<int>> voffset;
    voffset.push_back(offset);
    out_data->set_seq_offset(voffset);
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberSequenceConv, SequenceConvParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequenceConv, SequenceConvParam, X86, AK_INT8);

}
}
