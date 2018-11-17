
#include "saber_avx2_funcs.h"
#include "saber/funcs/impl/x86/saber_normal_activation.h"
#include "saber/funcs/debug.h"
#if defined(__AVX2__) and defined(__FMA__)
namespace anakin {

namespace saber {

void avx2_vector_softmax(const float* in, int length, float* out) {
    float max = _m256_max_array(in, length);
    __m256 max_vec = _mm256_set1_ps(max);
    __m256 exp_sum = _mm256_setzero_ps();
    int remainder = length % 8;
    int round_length = length / 8 * 8;

    if (remainder > 0) {
        for (int j = 0; j < round_length; j += 8) {
            __m256 temp_in = _mm256_loadu_ps(&in[j]);
            __m256 temp_exp = exp256_ps_fma(temp_in - max_vec);
            exp_sum += temp_exp;
            _mm256_storeu_ps(&out[j], temp_exp);
        }

        __m256i vec_mask = _m256_continue_mask_m256i(remainder);
        __m256 vec_mask_m256 = _m256_continue_mask_m256(remainder);
        __m256 temp_in = _mm256_maskload_ps(&in[round_length], vec_mask);
        __m256 temp_exp = _mm256_blendv_ps(_mm256_setzero_ps(), exp256_ps_fma(temp_in - max_vec),
                                           vec_mask_m256);
        _mm256_maskstore_ps(&out[round_length], vec_mask, temp_exp);
        exp_sum += temp_exp;

        float sum = _m256_self_sum(exp_sum);
        __m256 sum_vec = _mm256_set1_ps(sum);

        for (int j = 0; j < round_length; j += 8) {
            __m256 temp_in = _mm256_loadu_ps(&out[j]);
            _mm256_storeu_ps(&out[j], temp_in / sum_vec);
        }

        temp_in = _mm256_maskload_ps(&out[round_length], vec_mask);
        _mm256_maskstore_ps(&out[round_length], vec_mask, temp_in / sum_vec);

    } else {
        for (int j = 0; j < round_length; j += 8) {
            __m256 temp_in = _mm256_loadu_ps(&in[j]);
            __m256 temp_exp = exp256_ps_fma(temp_in - max_vec);
            exp_sum += temp_exp;
            _mm256_storeu_ps(&out[j], temp_exp);
        }

        float sum = _m256_self_sum(exp_sum);
        __m256 sum_vec = _mm256_set1_ps(sum);

        for (int j = 0; j < round_length; j += 8) {
            __m256 temp_in = _mm256_loadu_ps(&out[j]);
            _mm256_storeu_ps(&out[j], temp_in / sum_vec);
        }
    }

}

void avx2_sequence_softmax(const float* data, std::vector<int>& seq_offset, float* out) {
    for (int i = 0; i < seq_offset.size() - 1; i++) {
        int start = seq_offset[i];
        int end = seq_offset[i + 1];
        int length = end - start;
        const float* seq_in = &data[start];
        float* seq_out = &out[start];
        avx2_vector_softmax(seq_in, length, seq_out);
    }
}

void avx2_lstm_bias_and_act(const float* hidden_in, const float* bias_data, float* out,
                            float* cell_data, const int seq_num, const int hidden_size, const int with_peephole) {
    const float* bias_i = (float*)(bias_data);
    const float* bias_f = (float*)(bias_data + hidden_size);
    const float* bias_c = (float*)(bias_data + 2 * hidden_size);
    const float* bias_o = (float*)(bias_data + 3 * hidden_size);

    if (with_peephole) {
        const float* w_ci = (float*)(bias_data + 4 * hidden_size);
        const float* w_cf = (float*)(bias_data + 5 * hidden_size);
        const float* w_co = (float*)(bias_data + 6 * hidden_size);

        for (int i = 0; i < seq_num; i++) {
            const float*  tmp_hidden_i = (float*)(hidden_in + i * 4 * hidden_size);
            const float*  tmp_hidden_f = (float*)(hidden_in + i * 4 * hidden_size + 1 * hidden_size);
            const float*  tmp_hidden_c = (float*)(hidden_in + i * 4 * hidden_size + 2 * hidden_size);
            const float*  tmp_hidden_o = (float*)(hidden_in + i * 4 * hidden_size + 3 * hidden_size);
            float*  tmp_cell_data = (float*)(cell_data + i * hidden_size);
            float*  tmp_out = (float*)(out + i * hidden_size);

            for (int j = 0; j < hidden_size / 8; j++) {
                float ig = Sigmoid(tmp_hidden_i[j] + bias_i[j] + tmp_cell_data[j] * w_ci[j]);
                float fg = Sigmoid(tmp_hidden_f[j] + bias_f[j] + tmp_cell_data[j] * w_cf[j]);
                float c_t_0 = Tanh(tmp_hidden_c[j] + bias_c[j]);
                tmp_cell_data[j] = ig * c_t_0 + fg * tmp_cell_data[j];
                float og = Sigmoid(tmp_hidden_o[j] + bias_o[j] + tmp_cell_data[j] * w_co[j]);
                tmp_out[j] = og * Tanh(tmp_cell_data[j]);
            }
        }
    } else {
        for (int i = 0; i < seq_num; i++) {

            const float* tmp_hidden_i = hidden_in + i * 4 * hidden_size;
            const float* tmp_hidden_f = tmp_hidden_i + hidden_size;
            const float* tmp_hidden_c = tmp_hidden_f + hidden_size;
            const float* tmp_hidden_o = tmp_hidden_c + hidden_size;
            float* tmp_cell_data = cell_data + i * hidden_size;
            float* tmp_out = out + i * hidden_size;
            int round_hidden = hidden_size / 8 * 8;

            for (int j = 0; j < round_hidden; j += 8) {
                __m256 hidden = _mm256_loadu_ps(&tmp_hidden_i[j]);
                __m256 bias = _mm256_loadu_ps(&bias_i[j]);
                __m256 ig = Sigmoid(hidden + bias);
                hidden = _mm256_loadu_ps(&tmp_hidden_f[j]);
                bias = _mm256_loadu_ps(&bias_f[j]);
                __m256 fg = Sigmoid(hidden + bias);
                hidden = _mm256_loadu_ps(&tmp_hidden_o[j]);
                bias = _mm256_loadu_ps(&bias_o[j]);
                __m256 og = Sigmoid(hidden + bias);
                hidden = _mm256_loadu_ps(&tmp_hidden_c[j]);
                bias = _mm256_loadu_ps(&bias_c[j]);
                __m256 temp = hidden + bias;
                __m256 c_t_0 = Tanh(hidden + bias);
                __m256 cell = _mm256_loadu_ps(&tmp_cell_data[j]);
                cell = ig * c_t_0 + fg * cell;
                _mm256_storeu_ps(&tmp_cell_data[j], cell);
                _mm256_storeu_ps(&tmp_out[j], og * Tanh(cell));
            }

            int remainder = hidden_size % 8;

            if (remainder > 0) {
                __m256i _vec_mask = _m256_continue_mask_m256i(remainder);
                __m256 hidden = _mm256_maskload_ps(&tmp_hidden_i[round_hidden], _vec_mask);
                __m256 bias = _mm256_maskload_ps(&bias_i[round_hidden], _vec_mask);
                __m256 ig = Sigmoid(hidden + bias);
                hidden = _mm256_maskload_ps(&tmp_hidden_f[round_hidden], _vec_mask);
                bias = _mm256_maskload_ps(&bias_f[round_hidden], _vec_mask);
                __m256 fg = Sigmoid(hidden + bias);
                hidden = _mm256_maskload_ps(&tmp_hidden_o[round_hidden], _vec_mask);
                bias = _mm256_maskload_ps(&bias_o[round_hidden], _vec_mask);
                __m256 og = Sigmoid(hidden + bias);
                hidden = _mm256_maskload_ps(&tmp_hidden_c[round_hidden], _vec_mask);
                bias = _mm256_maskload_ps(&bias_c[round_hidden], _vec_mask);
                __m256 temp = hidden + bias;
                __m256 c_t_0 = Tanh(hidden + bias);
                __m256 cell = _mm256_maskload_ps(&tmp_cell_data[round_hidden], _vec_mask);

                //                printf_intrin_var(ig);
                //                printf_intrin_var(c_t_0);
                //                printf_intrin_var(fg);
                //                printf_intrin_var(cell);
                //                for (int j = 0; j < hidden_size; j++) {
                //                    printf("%f,",tmp_cell_data[j]);
                //                }
                //                printf("\n");
                cell = ig * c_t_0 + fg * cell;
                _mm256_maskstore_ps(&tmp_cell_data[round_hidden], _vec_mask, cell);
                _mm256_maskstore_ps(&tmp_out[round_hidden], _vec_mask, og * Tanh(cell));
            }

            //            exit(0);
        }
    }
}

void avx2_sequence_pool(const float* data, const float* weight, std::vector<int>& seq_offset,
                        int dim,
                        float* out) {
    int round_dim = dim / 8 * 8;
    int remainder = dim % 8;
    __m256i mask_m256i = _m256_continue_mask_m256i(remainder);

    for (int i = 0; i < seq_offset.size() - 1; i++) {
        for (int k = 0; k < round_dim; k += 8) {
            __m256 temp_out = _mm256_setzero_ps();

            for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
                float scale = weight[j];
                __m256 temp_scale = _mm256_set1_ps(scale);
                const float* tmp_data = data + j * dim;
                __m256 temp_in = _mm256_loadu_ps(&tmp_data[k]);
                temp_out += temp_in * temp_scale;
            }

            _mm256_storeu_ps(out +  i * dim + k, temp_out);
        }

        if (remainder > 0) {
            __m256 temp_out = _mm256_setzero_ps();

            for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
                float scale = weight[j];
                __m256 temp_scale = _mm256_set1_ps(scale);
                const float* tmp_data = data + j * dim;
                __m256 temp_in = _mm256_maskload_ps(&tmp_data[round_dim], mask_m256i);
                temp_out += temp_in * temp_scale;
            }

            _mm256_maskstore_ps(out +  i * dim + round_dim, mask_m256i, temp_out);
        }
    }
}


}
}
#endif