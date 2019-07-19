#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX2_FUNCS_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX2_FUNCS_H


#include <vector>
#include "saber/funcs/impl/x86/kernel/jit_generator.h"
namespace anakin {

namespace saber {

inline bool avx2_is_compiled(){
#if defined(__AVX2__) and defined(__FMA__)
    return true;
#else
    return false;
#endif
};

inline bool avx2_can_used(){
    return avx2_is_compiled()&&jit::mayiuse(jit::avx2);
};
#if defined(__AVX2__) and defined(__FMA__)
void avx2_vector_softmax_stride(const float* in, int col, int row, float* out);
void avx2_vector_softmax(const float* in, int length, float* out);
void avx2_vector_relu(const float* in, int length, float* out);
void avx2_vector_sigmoid(const float* in, int length, float* out);
void avx2_sequence_softmax(const float* data, std::vector<int>& seq_offset, float* out);
void avx2_lstm_bias_and_act(const float* hidden_in, const float* bias_data, float* out,
                            float* cell_data, const int seq_num, const int hidden_size, const int with_peephole);

void avx2_sequence_pool(const float* data,
                        const float* weight,
                        std::vector<int>& seq_offset,
                        int dim,
                        float* out);

void avx2_vector_soft_sign(const float* in,
                           int length,
                           float* out);

/* Calculate the angle between two vectors
 * cos(theta) = a'b / (|a| * |b|)
 * output is cos(theta)
 * */
void avx2_cos_sim(const float* in_0,
                  const float* in_1,
                  const int num,
                  const int len,
                  const float epsilon,
                  float* out);

/* Calculate the sum of two vectors
 * y[i] +=  x[i]
 * */
void avx2_vector_sum(const float* in_0,
                  const int len,
                  float* out);

/* Calculate the sum of two vectors
 * z[i] =  x[i] + y[i]
 * */
void avx2_vector_sum(const float* in_0,
                     const float* in_1,
                     const int len,
                     float* out);

/* Calculate the sub of two vectors
 * z[i] =  x[i] - y[i]
 * */
void avx2_vector_sub(const float* in_0,
                     const float* in_1,
                     const int len,
                     float* out);

/* Calculate the product of two vectors
 * z[i] =  x[i] * y[i]
 * */
void avx2_vector_mul(const float* in_0,
                     const float* in_1,
                     const int len,
                     float* out);
#endif
}
}


#endif //ANAKIN_SABER_AVX2_FUNCS_H
