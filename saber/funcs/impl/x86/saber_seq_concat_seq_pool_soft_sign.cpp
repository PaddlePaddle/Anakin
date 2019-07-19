#include "anakin_thread.h"
#include "saber/funcs/impl/x86/saber_seq_concat_seq_pool_soft_sign.h"
#include "saber/funcs/impl/x86/saber_seq_concat_seq_pool_soft_sign.h"
#include "mkl.h"
#if defined(__AVX2__) and defined(__FMA__)
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#endif
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberSeqConcatSeqPoolSoftSign<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SeqConcatSeqPoolSoftSignParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    _emb_size = inputs[0]->valid_size() / inputs[0]->num();
    int seq_len = inputs[0]->get_seq_offset()[0].size() - 1;
    for (int i = 1; i < inputs.size(); i++) {
        int cur_emb_size = inputs[i]->valid_size() / inputs[i]->num();
        int cur_seq_len = inputs[i]->get_seq_offset()[0].size() - 1 ;
        CHECK_EQ(_emb_size, cur_emb_size) << "emb size must be the same";
        CHECK_EQ(seq_len, cur_seq_len) << "seq len  must be the same";
    }
    _buf = new OpDataType[anakin_get_num_procs() * _emb_size];
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSeqConcatSeqPoolSoftSign<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SeqConcatSeqPoolSoftSignParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSeqConcatSeqPoolSoftSign<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SeqConcatSeqPoolSoftSignParam<X86> &param) {
    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    int emb_size = inputs[0]->valid_size() / inputs[0]->num();
    for (int i = 1; i < inputs.size(); i++) {
        int cur_emb_size = inputs[i]->valid_size() / inputs[i]->num();
        int cur_seq_num = inputs[i]->get_seq_offset()[0].size() - 1 ;
        CHECK_EQ(emb_size, cur_emb_size) << "emb size must be the same";
        CHECK_EQ(seq_num, cur_seq_num) << "seq len  must be the same";
    }

    outputs[0]->reshape(Shape({seq_num, emb_size, 1, 1}, Layout_NCHW));
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
    std::vector<std::vector<int>> offset_vecs;
    for (int i = 0; i < inputs.size(); i++) {
        offset_vecs.push_back(inputs[i]->get_seq_offset()[0]);
    }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seq_num; i++) {
         auto tmp_out = output_data + i * emb_size;
         auto tmp_buf = _buf + anakin_get_thread_num() * emb_size;
         memset(tmp_buf, 0, sizeof(OpDataType) * emb_size);
         for (int j = 0; j < inputs.size(); j++) {
             const OpDataType *in_data = (const OpDataType*)inputs[j]->data();
             for (int k = offset_vecs[j][i]; k < offset_vecs[j][i + 1]; k++) {
                 auto tmp_in  = in_data + k * emb_size; 
//#if defined(__AVX2__) and defined(__FMA__)
//                 avx2_vector_sum(tmp_in, emb_size, tmp_buf);
//#else
//#pragma omp parallel for schedule(static)
                 for (int m = 0; m < emb_size; m++) {
                     tmp_buf[m] += tmp_in[m];
                 }
//#endif
             }
         }

//#if defined(__AVX2__) and defined(__FMA__)
//        avx2_vector_soft_sign(tmp_buf, emb_size, tmp_out);
//#else
//#pragma omp parallel for schedule(static)
       for (int m = 0; m < emb_size; m++) {
           auto data = tmp_buf[m];
           auto tmp = data > 0 ? data : -data;
           tmp_out[m]  = data / (1 + tmp);
       }
//#endif
    }

    return SaberSuccess;
}

template class SaberSeqConcatSeqPoolSoftSign<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSeqConcatSeqPoolSoftSign, SeqConcatSeqPoolSoftSignParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSeqConcatSeqPoolSoftSign, SeqConcatSeqPoolSoftSignParam, X86, AK_INT8);
}
} // namespace anakin
