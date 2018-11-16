#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX2_FUNCS_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX2_FUNCS_H
#if defined(__AVX2__) and defined(__FMA__)

#include <vector>
namespace anakin {

namespace saber {

void avx2_vector_softmax(const float* in, int length, float* out);
void avx2_sequence_softmax(const float* data, std::vector<int>& seq_offset, float* out);
void avx2_lstm_bias_and_act(const float* hidden_in, const float* bias_data, float* out,
                            float* cell_data, const int seq_num, const int hidden_size, const int with_peephole);
void avx2_sequence_pool(const float* data, const float* weight, std::vector<int>& seq_offset, int dim,
                        float* out);

}
}

#endif
#endif //ANAKIN_SABER_AVX2_FUNCS_H
