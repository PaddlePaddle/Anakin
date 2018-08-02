#include "saber/funcs/impl/x86/saber_sequence_conv.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "mkl_cblas.h"
namespace anakin {
namespace saber {

static void gemm(const bool TransA, const bool TransB, int m, int n, int k, const float alpha,
                 const float* a, const float* b, const float beta, float* c) {
    //    cout << "(" << m << "," << n << "," << k << ")" << endl;
    int lda = (!TransA/* == CblasNoTrans*/) ? k : m;
    int ldb = (!TransB/* == CblasNoTrans*/) ? n : k;
    CBLAS_TRANSPOSE cuTransA =
        (!TransA/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cuTransB =
        (!TransB/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    cblas_sgemm(CblasRowMajor, cuTransA, cuTransB, m, n, k, alpha, a, k, b, n, beta, c, n);
};

template <typename Dtype>
static void im2col_2d_ocf(const Dtype* in, int start,int stride, int pad_up, int pad_down, int kernel_size,
                          Dtype* out, int seq_length, int hidden_size) {
    for (int out_row = 0; out_row < seq_length; ++out_row) {
        for (int col = 0; col < kernel_size; ++col) {
            int index = out_row + col - pad_up+start;
            int out_index = (out_row * kernel_size + col) * hidden_size;

            if (index < 0 || index >= seq_length) {
                for (int hidden_index = 0; hidden_index < hidden_size; ++hidden_index){
                    out[out_index + hidden_index] = 0;
                }
            } else{
                for (int hidden_index = 0; hidden_index < hidden_size; ++hidden_index){
                    out[out_index + hidden_index] = in[index * hidden_size + hidden_index];
                }
            }
        }
    }
}

template <>
SaberStatus SaberSequenceConv<X86, AK_FLOAT>::dispatch(
    const std::vector<OpTensor*>& inputs,
    std::vector<OpTensor*>& outputs,
    SequenceConvParam<X86>& param) {
    CHECK_GE(param.padding_trainable,false)<<"not support padding_trainable";
    OpTensor* in_data = inputs[0];
    OpTensor* out_data = outputs[0];
    std::vector<std::vector<int>> offset_vec_vec = in_data->get_seq_offset();
    std::vector<int> offset = offset_vec_vec[offset_vec_vec.size()-1];
    out_data->set_seq_offset(offset_vec_vec);

    int word_num = offset[offset.size() - 1];
    utils::try_expand_tensor(_temp_im2col_tensor,word_num * param.filter_tensor->height());

    for (int i = 0; i < offset.size() - 1; ++i) {
        int start = offset[i];
        int seq_length = offset[i + 1] - offset[i];
        im2col_2d_ocf(static_cast<const OpDataType*>(in_data->data()) + _hidden_size * start, _word_start, param.context_stride, _up_pad, _down_pad,
                      param.context_length, static_cast<OpDataType*>(_temp_im2col_tensor.mutable_data()) + _hidden_kernel_size * start, seq_length,
                      _hidden_size);
    }

    gemm(false, false, word_num, _feature_size, _hidden_kernel_size, 1.f, static_cast<const OpDataType*>(_temp_im2col_tensor.data()),
         static_cast<const OpDataType*>(param.filter_tensor->data()), 0.f, static_cast<OpDataType*>(out_data->mutable_data()));

    return SaberSuccess;
}
template class SaberSequenceConv<X86, AK_FLOAT>;

}
}
