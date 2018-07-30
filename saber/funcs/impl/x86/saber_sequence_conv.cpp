#include "saber/funcs/impl/x86/saber_sequence_conv.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/tensor_op.h"
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
                    //                    printf("%d -> %d [%f]\n",index+hidden_index,out_index+hidden_index,in[index+hidden_index]);
                    out[out_index + hidden_index] = in[index * hidden_size + hidden_index];
                }
            }
        }
    }
}

template <>
SaberStatus SaberSequenceConv<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    SequenceConvParam<OpTensor>& param) {
    DataTensor_in* in_data = inputs[0];
    DataTensor_out* out_data = outputs[0];
    std::vector<int> offset = in_data->get_seq_offset();
    out_data->set_seq_offset(offset);

    int word_num = offset[offset.size() - 1];
    _temp_im2col_tensor.try_expand_size(word_num * param.filter_tensor->height());


    for (int i = 0; i < offset.size() - 1; ++i) {
        int start = offset[i];
        int seq_length = offset[i + 1] - offset[i];
        im2col_2d_ocf(in_data->data() + _hidden_size * start, param.context_stride, _up_pad, _down_pad,
                      param.context_length, _temp_im2col_tensor.mutable_data() + _hidden_kernel_size * start, seq_length,
                      _hidden_size);
    }

    //    printf("up and down %d,%d\n",_up_pad,_down_pad);
//    for (int i = 0; i < word_num * param.filter_tensor->height(); i++) {
//        printf("[%d] = %f\n", i, _temp_im2col_tensor.data()[i]);
//    }

    gemm(false, false, word_num, _feature_size, _hidden_kernel_size, 1.f, _temp_im2col_tensor.data(),
         param.filter_tensor->data(), 0.f, out_data->mutable_data());

    out_data->set_seq_offset(offset);
    return SaberSuccess;
}


}
}
