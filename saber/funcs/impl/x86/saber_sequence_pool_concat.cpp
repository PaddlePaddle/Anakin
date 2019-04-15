#include "saber/funcs/impl/x86/saber_sequence_pool_concat.h"
#include "saber/funcs/impl/x86/saber_avx2_expand.h"
#include "saber/funcs/impl/x86/saber_avx512_expand.h"
namespace anakin {
namespace saber {


template <>
SaberStatus SaberSequencePoolConcat<X86, AK_FLOAT>::create(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequencePoolConcatParam<X86>& param,
        Context<X86>& ctx) {
    return SaberSuccess;
};

template <>
SaberStatus SaberSequencePoolConcat<X86, AK_FLOAT>::init(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequencePoolConcatParam<X86>& param,
        Context<X86>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
};

#if defined(__AVX2__)
static void avx2_sequence_pool_sum_concat(const float* data, std::vector<int>& seq_offset,
        int dim,
        float* out) {
    int round_dim = dim / 8 * 8;
    int remainder = dim % 8;
    __m256i mask_m256i = _m256_continue_mask_m256i(remainder);

#pragma omp parallel for
    for (int i = 0; i < seq_offset.size() - 1; i++) {
        for (int k = 0; k < round_dim; k += 8) {
            __m256 temp_out = _mm256_setzero_ps();

            for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
                const float* tmp_data = data + j * dim;
                __m256 temp_in = _mm256_loadu_ps(&tmp_data[k]);
                temp_out += temp_in;
            }

            _mm256_storeu_ps(out +  i * dim + k, temp_out);
        }

        if (remainder > 0) {
            __m256 temp_out = _mm256_setzero_ps();

            for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
                const float* tmp_data = data + j * dim;
                __m256 temp_in = _mm256_maskload_ps(&tmp_data[round_dim], mask_m256i);
                temp_out += temp_in;
            }

            _mm256_maskstore_ps(out +  i * dim + round_dim, mask_m256i, temp_out);
        }
    }
}
#endif

#if defined(__AVX512F__)
static void avx512_sequence_pool_sum_concat(const float* data, std::vector<int>& seq_offset,
        int dim,
        float* out) {
    int round_dim = dim / 16 * 16;
    int remainder = dim % 16;
    __mmask16 remain_mask = __mm512_get_mask(remainder);
    const int seq_number = seq_offset.size() - 1;

    if (round_dim == 0) {

#pragma omp parallel for
        for (int i = 0; i < seq_number; i++) {
            __m512 temp_out = _mm512_setzero_ps();

            for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
                const float* tmp_data = data + j * dim;
                temp_out = _mm512_add_ps(temp_out, _mm512_mask_loadu_ps(temp_out, remain_mask, tmp_data));
            }

            _mm512_mask_storeu_ps(out +  i * dim, remain_mask, temp_out);
        }

    } else {
#pragma omp parallel for
        for (int i = 0; i < seq_number; i++) {
            for (int k = 0; k < round_dim; k += 16) {
                __m512 temp_out = _mm512_setzero_ps();

                for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
                    const float* tmp_data = data + j * dim;
                    __m512 temp_in = _mm512_loadu_ps(&tmp_data[k]);
                    temp_out += temp_in;
                }

                _mm512_storeu_ps(out + i * dim + k, temp_out);
            }

            if (remainder > 0) {
                __m512 temp_out = _mm512_setzero_ps();

                for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
                    const float* tmp_data = data + j * dim;
                    temp_out = _mm512_add_ps(temp_out, _mm512_mask_loadu_ps(temp_out, remain_mask,
                                             &tmp_data[round_dim]));
                }

                _mm512_mask_storeu_ps(out + i * dim + round_dim, remain_mask, temp_out);

            }
        }
    }
}
#endif

template <>
SaberStatus SaberSequencePoolConcat<X86, AK_FLOAT>::dispatch(const std::vector<Tensor<X86>*>&
        inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequencePoolConcatParam<X86>& param) {
    CHECK_GE(inputs[0]->get_seq_offset().size(), 1);
    SequencePoolParam<X86> seq_param = param.sequence_pool_param;
    auto seq_vec = inputs[0]->get_seq_offset()[0];
    int seq_num = seq_vec.back();
    float* input_ptr = static_cast<float*>(inputs[0]->data());
    float* output_ptr = static_cast<float*>(outputs[0]->data());

    int out_channel = inputs[0]->valid_size() / seq_num;

    if (seq_param.sequence_pool_type == Sequence_pool_sum) {

#if defined(__AVX512F__)
        avx512_sequence_pool_sum_concat(input_ptr, seq_vec, out_channel, output_ptr);
#elif defined(__AVX2__)
        avx2_sequence_pool_sum_concat(input_ptr, seq_vec, out_channel, output_ptr);
#else
        LOG(FATAL) << "not support for not open avx2";
#endif
    } else {
        LOG(FATAL) << "not support " << seq_param.sequence_pool_type;
    }

    return SaberSuccess;
};

DEFINE_OP_TEMPLATE(SaberSequencePoolConcat, SequencePoolConcatParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequencePoolConcat, SequencePoolConcatParam, X86, AK_INT8);

}
}