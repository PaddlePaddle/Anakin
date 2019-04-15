#include "saber_avx2_funcs.h"
#include "saber/funcs/impl/x86/saber_normal_activation.h"
#include "saber/funcs/debug.h"
#if defined(__AVX2__) and defined(__FMA__)
namespace anakin {

namespace saber {

inline __m256 avx2_load_mask(const float* in, int length) {
    __m256i vec_mask = _m256_continue_mask_m256i(length);
    return _mm256_maskload_ps(in, vec_mask);
}

inline void avx2_save_mask(__m256& in, float* out, int length) {
    __m256i vec_mask = _m256_continue_mask_m256i(length);
    _mm256_maskstore_ps(out, vec_mask, in);
}

void avx2_vector_relu(const float* in, int length, float* out) {
    int remainder = length % 8;
    int round_length = length / 8 * 8;
    __m256 zero = _mm256_setzero_ps();
    #pragma omp parallel for schedule(static)

    for (int i = 0; i < length; i += 8) {
        __m256 temp = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], _mm256_max_ps(zero, temp));
    }

    if (remainder > 0) {
        __m256i vec_mask = _m256_continue_mask_m256i(remainder);
        __m256 temp = _mm256_maskload_ps(&in[round_length], vec_mask);
        _mm256_maskstore_ps(&out[round_length], vec_mask, _mm256_max_ps(zero, temp));
    }

};

void avx2_vector_sigmoid(const float* in, int length, float* out) {
    int remainder = length % 8;
    int round_length = length / 8 * 8;
    #pragma omp parallel for schedule(static)

    for (int i = 0; i < length; i += 8) {
        __m256 temp = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], Sigmoid(temp));
    }

    if (remainder > 0) {
        __m256i vec_mask = _m256_continue_mask_m256i(remainder);
        __m256 temp = _mm256_maskload_ps(&in[round_length], vec_mask);
        _mm256_maskstore_ps(&out[round_length], vec_mask, Sigmoid(temp));
    }

};

void avx2_vector_soft_sign(const float* in, int length, float* out) {
    int remainder = length % 8;
    int round_length = length / 8 * 8;

    __m256 one = _mm256_set1_ps(1.f);
    __m256 zero = _mm256_setzero_ps();
    #pragma omp parallel for schedule(static)

    for (int i = 0; i < length; i += 8) {
        __m256 src = _mm256_loadu_ps(&in[i]);
        __m256 src_abs = _mm256_max_ps(src, -src);
        __m256 denominator = _mm256_add_ps(src_abs, one);
        _mm256_storeu_ps(&out[i], _mm256_div_ps(src, denominator));
    }

    if (remainder > 0) {
        __m256i vec_mask = _m256_continue_mask_m256i(remainder);
        __m256 src = _mm256_maskload_ps(&in[round_length], vec_mask);
        __m256 src_abs = _mm256_max_ps(src, -src);
        __m256 denominator = _mm256_add_ps(src_abs, one);
        _mm256_maskstore_ps(&out[round_length], vec_mask, _mm256_div_ps(src, denominator));
    }

};

void avx2_vector_softmax_stride(const float* in, int col, int row, float* out) {
    int remainder_col = col % 8;
    int round_col = col / 8 * 8;

    for (int col_id = 0; col_id < round_col; col_id += 8) {

        __m256 max_vec = _mm256_set1_ps(-1e20);

        for (int row_id = 0; row_id < row; row_id++) {
            __m256 temp_in = _mm256_loadu_ps(&in[row_id * col + col_id]);
            max_vec = _mm256_max_ps(max_vec, temp_in);
        }

        __m256 exp_sum = _mm256_setzero_ps();

        for (int row_id = 0; row_id < row; row_id++) {
            __m256 temp_in = _mm256_loadu_ps(&in[row_id * col + col_id]);
            __m256 temp_in_exp = exp256_ps_fma(temp_in - max_vec);
            exp_sum = _mm256_add_ps(exp_sum, temp_in_exp);
            _mm256_storeu_ps(&out[row_id * col + col_id], temp_in_exp);
        }

        __m256 exp_sum_rev = _mm256_div_ps(_mm256_set1_ps(1), exp_sum);

        for (int row_id = 0; row_id < row; row_id++) {
            __m256 temp_in = _mm256_loadu_ps(&out[row_id * col + col_id]);
            _mm256_storeu_ps(&out[row_id * col + col_id], _mm256_mul_ps(temp_in, exp_sum_rev));
        }
    }

    if (remainder_col > 0) {

        const __m256i vec_mask = _m256_continue_mask_m256i(remainder_col);
        __m256 max_vec = _mm256_set1_ps(-1e20);

        for (int row_id = 0; row_id < row; row_id++) {
            __m256 temp_in = _mm256_maskload_ps(&in[row_id * col + round_col], vec_mask);
            max_vec = _mm256_max_ps(max_vec, temp_in);
        }

        __m256 exp_sum = _mm256_setzero_ps();

        for (int row_id = 0; row_id < row; row_id++) {
            __m256 temp_in = _mm256_maskload_ps(&in[row_id * col + round_col], vec_mask);
            __m256 temp_in_exp = exp256_ps_fma(temp_in - max_vec);
            exp_sum = exp_sum + temp_in_exp;
            _mm256_maskstore_ps(&out[row_id * col + round_col], vec_mask, temp_in_exp);
        }

        __m256 exp_sum_rev = _mm256_div_ps(_mm256_set1_ps(1), exp_sum);

        for (int row_id = 0; row_id < row; row_id++) {
            __m256 temp_in = _mm256_maskload_ps(&out[row_id * col + round_col], vec_mask);
            _mm256_maskstore_ps(&out[row_id * col + round_col], vec_mask, _mm256_mul_ps(temp_in, exp_sum_rev));
        }
    }
}


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
        __m256 sum_vec = _mm256_set1_ps(1.f / sum);

        for (int j = 0; j < round_length; j += 8) {
            __m256 temp_in = _mm256_loadu_ps(&out[j]);
            _mm256_storeu_ps(&out[j], temp_in * sum_vec);
        }

        temp_in = _mm256_maskload_ps(&out[round_length], vec_mask);
        _mm256_maskstore_ps(&out[round_length], vec_mask, temp_in * sum_vec);

    } else {
        for (int j = 0; j < round_length; j += 8) {
            __m256 temp_in = _mm256_loadu_ps(&in[j]);
            __m256 temp_exp = exp256_ps_fma(temp_in - max_vec);
            exp_sum += temp_exp;
            _mm256_storeu_ps(&out[j], temp_exp);
        }

        float sum = _m256_self_sum(exp_sum);
        __m256 sum_vec = _mm256_set1_ps(1.f / sum);

        for (int j = 0; j < round_length; j += 8) {
            __m256 temp_in = _mm256_loadu_ps(&out[j]);
            _mm256_storeu_ps(&out[j], temp_in * sum_vec);
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

void avx2_cos_sim(const float* in_0,
                  const float* in_1,
                  const int num,
                  const int len,
                  const float epsilon,
                  float* out) {
    int round_dim = len / 8 * 8;
    int remainder = len % 8;
    __m256i mask_m256i = _m256_continue_mask_m256i(remainder);

    for (int n = 0; n < num; n++) {
        __m256 aa_sum = _mm256_setzero_ps();
        __m256 bb_sum = _mm256_setzero_ps();
        __m256 ab_sum = _mm256_setzero_ps();

        for (int k = 0; k < round_dim; k += 8) {
            __m256 a = _mm256_loadu_ps(&in_0[k]);
            __m256 b = _mm256_loadu_ps(&in_1[k]);
            aa_sum = _mm256_fmadd_ps(a, a, aa_sum);
            bb_sum = _mm256_fmadd_ps(b, b, bb_sum);
            ab_sum = _mm256_fmadd_ps(a, b, ab_sum);
        }

        if (remainder > 0) {
            __m256 a = _mm256_maskload_ps(&in_0[round_dim], mask_m256i);
            __m256 b = _mm256_maskload_ps(&in_1[round_dim], mask_m256i);
            aa_sum = _mm256_fmadd_ps(a, a, aa_sum);
            bb_sum = _mm256_fmadd_ps(b, b, bb_sum);
            ab_sum = _mm256_fmadd_ps(a, b, ab_sum);
        }

        float a_square_sum = _m256_self_sum(aa_sum);
        float b_square_sum = _m256_self_sum(bb_sum);
        float ab_prod_sum = _m256_self_sum(ab_sum);
        float c = a_square_sum * b_square_sum;

        if (c < epsilon) {
            out[n] = 0.f;
        } else {
            out[n] = ab_prod_sum / sqrt(c);
        }

        in_0 += len;
        in_1 += len;
    }

}

void avx2_vector_sum(const float* in_0,
                     const int len,
                     float* out) {
    int round_dim = len / 8 * 8;
    int remainder = len % 8;
    __m256i mask_m256i = _m256_continue_mask_m256i(remainder);
    #pragma omp parallel for schedule(static)

    for (int k = 0; k < round_dim; k += 8) {
        __m256 a = _mm256_loadu_ps(&in_0[k]);
        __m256 b = _mm256_loadu_ps(&out[k]);
        _mm256_storeu_ps(&out[k], _mm256_add_ps(a, b));
    }

    if (remainder > 0) {
        __m256 a = _mm256_maskload_ps(&in_0[round_dim], mask_m256i);
        __m256 b = _mm256_maskload_ps(&out[round_dim], mask_m256i);
        _mm256_maskstore_ps(out + round_dim, mask_m256i, _mm256_add_ps(a, b));
    }
}

void avx2_vector_sum(const float* in_0,
                     const float* in_1,
                     const int len,
                     float* out) {
    int round_dim = len / 8 * 8;
    int remainder = len % 8;
    __m256i mask_m256i = _m256_continue_mask_m256i(remainder);

    for (int k = 0; k < round_dim; k += 8) {
        __m256 a = _mm256_loadu_ps(&in_0[k]);
        __m256 b = _mm256_loadu_ps(&in_1[k]);
        _mm256_storeu_ps(&out[k], _mm256_add_ps(a, b));
    }

    if (remainder > 0) {
        __m256 a = _mm256_maskload_ps(&in_0[round_dim], mask_m256i);
        __m256 b = _mm256_maskload_ps(&in_1[round_dim], mask_m256i);
        _mm256_maskstore_ps(out + round_dim, mask_m256i, _mm256_add_ps(a, b));
    }
}

void avx2_vector_sub(const float* in_0,
                     const float* in_1,
                     const int len,
                     float* out) {
    int round_dim = len / 8 * 8;
    int remainder = len % 8;
    __m256i mask_m256i = _m256_continue_mask_m256i(remainder);

    for (int k = 0; k < round_dim; k += 8) {
        __m256 a = _mm256_loadu_ps(&in_0[k]);
        __m256 b = _mm256_loadu_ps(&in_1[k]);
        _mm256_storeu_ps(&out[k], _mm256_sub_ps(a, b));
    }

    if (remainder > 0) {
        __m256 a = _mm256_maskload_ps(&in_0[round_dim], mask_m256i);
        __m256 b = _mm256_maskload_ps(&in_1[round_dim], mask_m256i);
        _mm256_maskstore_ps(out + round_dim, mask_m256i, _mm256_sub_ps(a, b));
    }
}


void avx2_vector_mul(const float* in_0,
                     const float* in_1,
                     const int len,
                     float* out) {
    int round_dim = len / 8 * 8;
    int remainder = len % 8;
    __m256i mask_m256i = _m256_continue_mask_m256i(remainder);

    for (int k = 0; k < round_dim; k += 8) {
        __m256 a = _mm256_loadu_ps(&in_0[k]);
        __m256 b = _mm256_loadu_ps(&in_1[k]);
        _mm256_storeu_ps(&out[k], _mm256_mul_ps(a, b));
    }

    if (remainder > 0) {
        __m256 a = _mm256_maskload_ps(&in_0[round_dim], mask_m256i);
        __m256 b = _mm256_maskload_ps(&in_1[round_dim], mask_m256i);
        _mm256_maskstore_ps(out + round_dim, mask_m256i, _mm256_mul_ps(a, b));
    }
}

}
}
#endif
