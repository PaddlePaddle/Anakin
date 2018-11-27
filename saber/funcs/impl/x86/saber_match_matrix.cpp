#include "saber/funcs/impl/x86/saber_match_matrix.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberMatchMatrix<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        MatchMatrixParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberMatchMatrix<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        MatchMatrixParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    auto offset_l = inputs[0]->get_seq_offset()[0];
    auto offset_r = inputs[1]->get_seq_offset()[0];
    int batch = offset_r.size() - 1;
    int batch_word_r = offset_r[batch];
    int len_l = offset_l[1] - offset_l[0];
    int dim_t = param.dim_t;
    int dim_in = param.dim_in;
    int max_len_r = 0;
    for (int i = 0; i < offset_r.size() - 1; i++) {
        int cur_len = offset_r[i+1] - offset_r[i];
        max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
    }
    _input_l_transform.reshape(std::vector<int>{1, dim_t, dim_in, len_l});
    _input_l_transform_reorganize.reshape(std::vector<int>{1, dim_t, len_l, dim_in});
    _output_tmp.reshape(std::vector<int>{1, batch_word_r, dim_t, len_l});
    outputs[0]->reshape(std::vector<int>{batch, dim_t, len_l, max_len_r});
    
    return SaberSuccess;
}
template<typename dtype>
void transpose(const dtype* in, int M , int N , dtype* out) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
             out[j * M + i] = in[i * N + j];
        }
    }
}
/*(total_len_r, dim_t, len_l)->(batch, dim_t, len_l, max_len_r)*/
template<typename dtype>
void padding_out(const dtype* src, std::vector<int>& offset_r, int dim_t, int len_l, dtype* dst) {
    int max_len_r = 0;
    for (int i = 0; i < offset_r.size() - 1; i++) {
        int cur_len = offset_r[i+1] - offset_r[i];
        max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
    }
    int seq_num  = offset_r.size() - 1;
    int tl = dim_t * len_l;
    for (int i = 0; i < seq_num; i++) {
        dtype* dst_tmp = dst + i * tl * max_len_r;
        const dtype* src_tmp = src + offset_r[i] *  tl;
        int cur_len = offset_r[i+1] - offset_r[i];
        for (int j = 0; j < cur_len; j++) {
            for (int k = 0; k < tl; k++) {
                dst_tmp[k * max_len_r + j] = src_tmp[j * tl + k];
            }
        }
        for (int k = 0; k < tl; k++) {
            memset(dst_tmp + k * max_len_r + cur_len, 0, sizeof(dtype) * (max_len_r - cur_len));
        }
    }
}
template <DataType OpDtype>
SaberStatus SaberMatchMatrix<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        MatchMatrixParam<X86> &param) {
    CHECK_EQ(inputs.size(), 2) <<"topk pooling need two inputs";
    int dim_t = param.dim_t;
    int dim_in = param.dim_in;
    auto offset_l = inputs[0]->get_seq_offset()[0];
    auto offset_r = inputs[1]->get_seq_offset()[0];
    int len_l = offset_l[1] - offset_l[0];
    int len_r = offset_r[offset_r.size() - 1];
    const OpDataType* weight_data =  (const OpDataType*) param.weight()->data();
    const OpDataType* input_l = (const OpDataType*)inputs[0]->data();
    const OpDataType* input_r = (const OpDataType*)inputs[1]->data();
    OpDataType* input_l_transform = (OpDataType*)_input_l_transform.mutable_data();
    OpDataType* input_l_transform_reorganize = (OpDataType*)_input_l_transform_reorganize.mutable_data();
    OpDataType* output_tmp = (OpDataType*)_output_tmp.mutable_data();
    OpDataType* output_data = (OpDataType*) outputs[0]->mutable_data();
    _gemm_l_transform.init(true, true, dim_t * dim_in, len_l, dim_in, *(this->_ctx));
    _gemm_l_transform.dispatch(1.0f, 0.f, weight_data, input_l,  input_l_transform);
    for (int i = 0; i < dim_t; i++) {
        int offset =  i * dim_in * len_l;
        transpose<OpDataType>(input_l_transform + offset, dim_in, len_l, input_l_transform_reorganize +  offset);
    }
    _gemm_r_transform.init(false, true, len_r, dim_t*len_l, dim_in, *(this->_ctx));
    _gemm_r_transform.dispatch(1.0f, 0.f, input_r, input_l_transform_reorganize, output_tmp);
    padding_out(output_tmp, offset_r, dim_t, len_l, output_data);
    outputs[0]->set_seq_offset(inputs[1]->get_seq_offset());
    
    return SaberSuccess;
}

template class SaberMatchMatrix<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberMatchMatrix, MatchMatrixParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberMatchMatrix, MatchMatrixParam, X86, AK_INT8);
}
} // namespace anakin
