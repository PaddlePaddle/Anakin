
#include "saber/funcs/impl/x86/intrinsic_packed_fc.h"
#include <x86intrin.h>
#include "jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "debug.h"
namespace anakin {
namespace saber {
namespace jit {

#define USE_OMP_IN_INTRINSIC_PACKED_FC 0

#define GET_OFF(field) offsetof(jit_int8_packed_fc_call_t, field)
using namespace Xbyak;

void jit_s8s8s32_packed_gemm::cal_one_block() {
    /**
        ma0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
     */
    vpmovsxbw(a0, ptr[address_a_0]);
    vpmovsxbw(a1, ptr[address_a_1]);
    vpmovsxbw(b0, ptr[address_b_0]);
    vpmovsxbw(b1, ptr[address_b_1]);
    vpmovsxbw(b2, ptr[address_b_2]);
    vpmovsxbw(b3, ptr[address_b_3]);
    /**
            temp_0 = _mm256_madd_epi16(ma0, mb0);
            temp_1 = _mm256_madd_epi16(ma1, mb0);
            sum0 = _mm256_add_epi32(sum0, temp_0);
            sum1 = _mm256_add_epi32(sum1, temp_1);
     */

    vpmaddwd(vtemp_0, a0, b0);
    vpmaddwd(vtemp_1, a1, b0);
    vpaddd(sum_row0_col0, vtemp_0, sum_row0_col0);
    vpaddd(sum_row1_col0, vtemp_1, sum_row1_col0);

    add(address_a_0, reg_k_block_size);
    add(address_a_1, reg_k_block_size);
    add(address_b_0, reg_k_block_size);
    add(address_b_1, reg_k_block_size);
    add(address_b_2, reg_k_block_size);
    add(address_b_3, reg_k_block_size);

    vpmaddwd(vtemp_0, a0, b1);
    vpmaddwd(vtemp_1, a1, b1);
    vpaddd(sum_row0_col1, vtemp_0, sum_row0_col1);
    vpaddd(sum_row1_col1, vtemp_1, sum_row1_col1);


    vpmaddwd(vtemp_0, a0, b2);
    vpmaddwd(vtemp_1, a1, b2);
    vpaddd(sum_row0_col2, vtemp_0, sum_row0_col2);
    vpaddd(sum_row1_col2, vtemp_1, sum_row1_col2);


    vpmaddwd(vtemp_0, a0, b3);
    vpmaddwd(vtemp_1, a1, b3);
    vpaddd(sum_row0_col3, vtemp_0, sum_row0_col3);
    vpaddd(sum_row1_col3, vtemp_1, sum_row1_col3);

}

void jit_s8s8s32_packed_gemm::load_and_init() {
    mov(reg_lda, ptr[this->param1 + GET_OFF(lda)]);
    mov(reg_ldb, ptr[this->param1 + GET_OFF(ldb)]);
    /**
     *
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;
     */
    mov(address_a_0, reg_input);
    mov(address_a_1, reg_input);
    add(address_a_1, reg_lda);
    mov(reg_ldc, ptr[this->param1 + GET_OFF(ldc)]);

    mov(address_b_0, reg_weights);
    mov(address_b_1, reg_weights);
    add(address_b_1, reg_ldb);
    mov(address_b_2, address_b_1);
    add(address_b_2, reg_ldb);
    mov(address_b_3, address_b_2);
    add(address_b_3, reg_ldb);

    vpxor(sum_row0_col0, sum_row0_col0, sum_row0_col0);
    vpxor(sum_row1_col0, sum_row1_col0, sum_row1_col0);
    vpxor(sum_row0_col1, sum_row0_col1, sum_row0_col1);
    vpxor(sum_row1_col1, sum_row1_col1, sum_row1_col1);
    vpxor(sum_row0_col2, sum_row0_col2, sum_row0_col2);
    vpxor(sum_row1_col2, sum_row1_col2, sum_row1_col2);
    vpxor(sum_row0_col3, sum_row0_col3, sum_row0_col3);
    vpxor(sum_row1_col3, sum_row1_col3, sum_row1_col3);

}

void jit_s8s8s32_packed_gemm::reduction_and_store2mem() {
    vpxor(zero_in_reduction, zero_in_reduction, zero_in_reduction);
    /**
    *
    sum0 = _mm256_hadd_epi32(sum0, sum2);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum4 = _mm256_hadd_epi32(sum4, sum6);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc0[2] = _mm256_extract_epi32(sum4, 0);
    pc0[3] = _mm256_extract_epi32(sum4, 1);
    */

    vphaddd(c_row0_col0_1, sum_row0_col0, sum_row0_col1);
    vphaddd(c_row0_col0_1, c_row0_col0_1, zero_in_reduction);
    vperm2i128(temp0_in_reduction, c_row0_col0_1, zero_in_reduction, 0x31);
    vpaddd(c_row0_col0_1, temp0_in_reduction, c_row0_col0_1);


    vphaddd(c_row0_col2_3, sum_row0_col2, sum_row0_col3);
    vphaddd(c_row0_col2_3, c_row0_col2_3, zero_in_reduction);
    vperm2i128(temp1_in_reduction, c_row0_col2_3, zero_in_reduction, 0x31);
    vpaddd(c_row0_col2_3, temp1_in_reduction, c_row0_col2_3);

    vpermq(c_row0_col2_3, c_row0_col2_3, 0x00);
    vpblendd(c_row0_col0_1_2_3, c_row0_col0_1, c_row0_col2_3, 0x0c);
    movdqu(ptr[reg_output], c_row0_col0_1_2_3_m128);
    /**
     *
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum1, 0);
    pc1[1] = _mm256_extract_epi32(sum1, 1);

    //the 3 row
    sum5 = _mm256_hadd_epi32(sum5, sum7);
    sum5 = _mm256_hadd_epi32(sum5, zero);
    sum5 = _mm256_add_epi32(sum5, _mm256_permute2x128_si256(sum5, zero, 0x31));

    pc1[2] = _mm256_extract_epi32(sum5, 0);
    pc1[3] = _mm256_extract_epi32(sum5, 1);
     */

    vphaddd(c_row1_col0_1, sum_row1_col0, sum_row1_col1);
    vphaddd(c_row1_col0_1, c_row1_col0_1, zero_in_reduction);
    vperm2i128(temp2_in_reduction, c_row1_col0_1, zero_in_reduction, 0x31);
    vpaddd(c_row1_col0_1, temp2_in_reduction, c_row1_col0_1);

    vphaddd(c_row1_col2_3, sum_row1_col2, sum_row1_col3);
    vphaddd(c_row1_col2_3, c_row1_col2_3, zero_in_reduction);
    vperm2i128(temp3_in_reduction, c_row1_col2_3, zero_in_reduction, 0x31);
    vpaddd(c_row1_col2_3, temp3_in_reduction, c_row1_col2_3);


    vpermq(c_row1_col2_3, c_row1_col2_3, 0x00);
    vpblendd(c_row1_col0_1_2_3, c_row1_col0_1, c_row1_col2_3, 0x0c);

    mov(rax, 4);
    mul(reg_ldc);
    add(reg_output, rax);
    movdqu(ptr[reg_output], c_row1_col0_1_2_3_m128);
}

/*void jit_s8s8s32_packed_gemm::generate() {
    this->preamble();
    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_weights, ptr[this->param1 + GET_OFF(weights)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(output_data)]);
    mov(reg_k_block_num, ptr[this->param1 + GET_OFF(k_block)]);
    mov(reg_k_block_size, aligned_length);

    load_and_init();

    L("FOR_01");
    cal_one_block();

    dec(reg_k_block_num);
    jnz("FOR_01");

    reduction_and_store2mem();

    this->postamble();
}*/


void jit_s8s8s32_packed_gemm::generate() {
    this->preamble();
    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_weights, ptr[this->param1 + GET_OFF(weights)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(output_data)]);
    mov(reg_k_block_num, ptr[this->param1 + GET_OFF(k_block)]);
    mov(reg_k_block_size, aligned_length);

    mov(reg_lda, ptr[this->param1 + GET_OFF(lda)]);
    mov(reg_ldb, ptr[this->param1 + GET_OFF(ldb)]);

    mov(address_a_0, reg_input);
    vpmovsxbw(a0, ptr[address_a_0]);
    mov(address_a_1, reg_input);
    add(address_a_1, reg_lda);
    mov(reg_ldc, ptr[this->param1 + GET_OFF(ldc)]);
    vpmovsxbw(a1, ptr[address_a_1]);

    mov(address_b_0, reg_weights);
    vpmovsxbw(b0, ptr[address_b_0]);
    mov(address_b_1, reg_weights);
    add(address_b_1, reg_ldb);
    vpmovsxbw(b1, ptr[address_b_1]);
    mov(address_b_2, address_b_1);
    add(address_b_2, reg_ldb);
    vpmovsxbw(b2, ptr[address_b_2]);
    mov(address_b_3, address_b_2);
    add(address_b_3, reg_ldb);
    vpmovsxbw(b3, ptr[address_b_3]);

    vpxor(sum_row0_col0, sum_row0_col0, sum_row0_col0);
    vmovdqa(sum_row1_col0, sum_row0_col0);
    vmovdqa(sum_row0_col1, sum_row0_col0);
    vmovdqa(sum_row1_col1, sum_row0_col0);
    vmovdqa(sum_row0_col2, sum_row0_col0);
    vmovdqa(sum_row1_col2, sum_row0_col0);
    vmovdqa(sum_row0_col3, sum_row0_col0);
    vmovdqa(sum_row1_col3, sum_row0_col0);

    //    LOG(INFO)<<"jcp.k_block_number "<<jcp.k_block_number;
    for (int i = 0; i < jcp.k_block_number; i++) {
        if (i != 0) {
            vpmovsxbw(b0, ptr[address_b_0]);
            vpmovsxbw(b1, ptr[address_b_1]);
            vpmovsxbw(b2, ptr[address_b_2]);
            vpmovsxbw(b3, ptr[address_b_3]);
        }

        vpmaddwd(vtemp_0, a0, b0);
        vpmaddwd(vtemp_1, a1, b0);
        vpaddd(sum_row0_col0, vtemp_0, sum_row0_col0);
        vpaddd(sum_row1_col0, vtemp_1, sum_row1_col0);



        vpmaddwd(vtemp_0, a0, b1);
        vpmaddwd(vtemp_1, a1, b1);
        vpaddd(sum_row0_col1, vtemp_0, sum_row0_col1);
        vpaddd(sum_row1_col1, vtemp_1, sum_row1_col1);


        vpmaddwd(vtemp_0, a0, b2);
        vpmaddwd(vtemp_1, a1, b2);
        vpaddd(sum_row0_col2, vtemp_0, sum_row0_col2);
        vpaddd(sum_row1_col2, vtemp_1, sum_row1_col2);


        vpmaddwd(vtemp_0, a0, b3);
        vpmaddwd(vtemp_1, a1, b3);
        vpaddd(sum_row0_col3, vtemp_0, sum_row0_col3);
        vpaddd(sum_row1_col3, vtemp_1, sum_row1_col3);

        add(address_a_0, reg_k_block_size);
        add(address_a_1, reg_k_block_size);
        vpmovsxbw(a0, ptr[address_a_0]);
        vpmovsxbw(a1, ptr[address_a_1]);
        add(address_b_0, reg_k_block_size);
        add(address_b_1, reg_k_block_size);
        add(address_b_2, reg_k_block_size);
        add(address_b_3, reg_k_block_size);

    }


    reduction_and_store2mem();

    this->postamble();
}


}
}
}

namespace anakin {
namespace saber {

#if defined(__AVX2__)

inline __m256i load_int8_to_int16(const void* ptr) {
    return _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) ptr));
}
inline void load_2int16_madd(const int& epi16x2, const __m256i& b, __m256i& c) {
    c = _mm256_add_epi32(c, _mm256_madd_epi16(_mm256_set1_epi32(epi16x2), b));
}

void packed_weights_k2(Tensor<X86>& inner_tensor, const Tensor<X86>& weights_tensor, const int n,
                       const int k, int slice_n) {
    CHECK_EQ(weights_tensor.get_dtype(), AK_INT8);
    CHECK_EQ(k % 2, 0) << "only support k % 16 = 0";
    CHECK_EQ(n % slice_n, 0) << "only support n % 8 = 0";
    const int new_row = n / slice_n;
    const int new_col = k * slice_n;
    inner_tensor.re_alloc(Shape({1, 1, new_row, new_col}), weights_tensor.get_dtype());
    const int8_t* in_ptr = static_cast<int8_t*>(weights_tensor.data());
    int8_t* out_ptr = static_cast<int8_t*>(inner_tensor.data());

    for (int row = 0; row < k; row++) {
        for (int col = 0; col < n; col++) {
            int out_row = col / slice_n;
            int slice_id = row / 2;
            int slice_inner_id_0 = row % 2;
            int slice_inner_id_1 = col % slice_n;
            int output_index = out_row * new_col + slice_id * 2 * slice_n + slice_inner_id_1 * 2 +
                               slice_inner_id_0;
            int input_index = row * n + col;
            out_ptr[output_index] = in_ptr[input_index];
        }
    }

    Tensor<X86>temp_tensor = weights_tensor;
}

void packed_weights_k2_split_k(Tensor<X86>& inner_tensor, const Tensor<X86>& weights_tensor,
                               const int n, const int k, int slice_n, int slice_n_inner_length) {
    CHECK_EQ(weights_tensor.get_dtype(), AK_INT8);
    CHECK_EQ(k % (2 * 8), 0) << "only support k % 16 = 0";
    CHECK_EQ(n % 8, 0) << "only support n % 8 = 0";
    const int new_row = n / slice_n;
    const int new_col = k * slice_n;
    inner_tensor.re_alloc(Shape({1, 1, new_row, new_col}), weights_tensor.get_dtype());
    const int8_t* in_ptr = static_cast<int8_t*>(weights_tensor.data());
    int8_t* out_ptr = static_cast<int8_t*>(inner_tensor.data());

    for (int row = 0; row < k; row++) {
        for (int col = 0; col < n; col++) {
            int out_row = col / slice_n;
            int slice_id = row / 2;
            int slice_inner_id_0 = row % 2;
            int slice_inner_id_1 = col % slice_n;
            int output_index = out_row * new_col + slice_id * 2 * slice_n + slice_inner_id_1 * 2 +
                               slice_inner_id_0;
            int input_index = row * n + col;
            out_ptr[output_index] = in_ptr[input_index];
        }
    }

    Tensor<X86>temp_tensor = weights_tensor;
}

void packed_weights_transpose_k(Tensor<X86>& inner_tensor, const Tensor<X86>& weights_tensor,
                                const int n, const int k,
                                const int n_slice, const int k_slice) {
    CHECK_EQ(weights_tensor.get_dtype(), AK_INT8);
    CHECK_EQ(k % 16, 0) << "only support k % 16 = 0";
    CHECK_EQ(n % n_slice, 0) << "only support n % 8 = 0";
    const int new_row = n / n_slice;
    const int new_col = k * n_slice;
    inner_tensor.re_alloc(Shape({1, 1, new_row, new_col}), weights_tensor.get_dtype());
    const int8_t* in_ptr = static_cast<int8_t*>(weights_tensor.data());
    int8_t* out_ptr = static_cast<int8_t*>(inner_tensor.data());

    for (int row = 0; row < k; row++) {
        for (int col = 0; col < n; col++) {
            int out_row = col / n_slice;
            int slice_id = row / k_slice;
            int slice_inner_id_0 = row % k_slice;
            int slice_inner_id_1 = col % n_slice;
            int output_index = out_row * new_col + slice_id * k_slice * n_slice + slice_inner_id_1 * k_slice +
                               slice_inner_id_0;
            int input_index = row * n + col;
            out_ptr[output_index] = in_ptr[input_index];
        }
    }
}

void block4x128_kernel_avx2_me(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {

    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * 16;
    const int8_t* pb2 = pb0 + 2 * 16;
    const int8_t* pb3 = pb0 + 3 * 16;
    const int8_t* pb4 = pb0 + 4 * 16;
    const int8_t* pb5 = pb0 + 5 * 16;
    const int8_t* pb6 = pb0 + 6 * 16;
    const int8_t* pb7 = pb0 + 7 * 16;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 4;
    size_t k_leftover = k - (nk << 4);


    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256();
    __m256i c3 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        const __m256i b0 = load_int8_to_int16(pb0);
        const __m256i b1 = load_int8_to_int16(pb1);
        const __m256i b2 = load_int8_to_int16(pb2);
        const __m256i b3 = load_int8_to_int16(pb3);
        const __m256i b4 = load_int8_to_int16(pb4);
        const __m256i b5 = load_int8_to_int16(pb5);
        const __m256i b6 = load_int8_to_int16(pb6);
        const __m256i b7 = load_int8_to_int16(pb7);

        const __v8si a0 = (__v8si)load_int8_to_int16(pa0);
        const __v8si a1 = (__v8si)load_int8_to_int16(pa1);
        const __v8si a2 = (__v8si)load_int8_to_int16(pa2);
        const __v8si a3 = (__v8si)load_int8_to_int16(pa3);

        load_2int16_madd(a0[0], b0, c0);
        load_2int16_madd(a0[1], b1, c0);
        load_2int16_madd(a0[2], b2, c0);
        load_2int16_madd(a0[3], b3, c0);
        load_2int16_madd(a0[4], b4, c0);
        load_2int16_madd(a0[5], b5, c0);
        load_2int16_madd(a0[6], b6, c0);
        load_2int16_madd(a0[7], b7, c0);

        load_2int16_madd(a1[0], b0, c1);
        load_2int16_madd(a1[1], b1, c1);
        load_2int16_madd(a1[2], b2, c1);
        load_2int16_madd(a1[3], b3, c1);
        load_2int16_madd(a1[4], b4, c1);
        load_2int16_madd(a1[5], b5, c1);
        load_2int16_madd(a1[6], b6, c1);
        load_2int16_madd(a1[7], b7, c1);

        load_2int16_madd(a2[0], b0, c2);
        load_2int16_madd(a2[1], b1, c2);
        load_2int16_madd(a2[2], b2, c2);
        load_2int16_madd(a2[3], b3, c2);
        load_2int16_madd(a2[4], b4, c2);
        load_2int16_madd(a2[5], b5, c2);
        load_2int16_madd(a2[6], b6, c2);
        load_2int16_madd(a2[7], b7, c2);

        load_2int16_madd(a3[0], b0, c3);
        load_2int16_madd(a3[1], b1, c3);
        load_2int16_madd(a3[2], b2, c3);
        load_2int16_madd(a3[3], b3, c3);
        load_2int16_madd(a3[4], b4, c3);
        load_2int16_madd(a3[5], b5, c3);
        load_2int16_madd(a3[6], b6, c3);
        load_2int16_madd(a3[7], b7, c3);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;

        pb0 += 16 * 8;
        pb1 += 16 * 8;
        pb2 += 16 * 8;
        pb3 += 16 * 8;
        pb4 += 16 * 8;
        pb5 += 16 * 8;
        pb6 += 16 * 8;
        pb7 += 16 * 8;

    }

    _mm256_storeu_si256((__m256i*)pc0, c0);
    _mm256_storeu_si256((__m256i*)pc1, c1);
    _mm256_storeu_si256((__m256i*)pc2, c2);
    _mm256_storeu_si256((__m256i*)pc3, c3);
}
void block_mx8_kernel_avx2_me(const int32_t m,
                              const int32_t k, const int8_t* a, const int32_t lda,
                              const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * 16;
    const int8_t* pb2 = pb0 + 2 * 16;
    const int8_t* pb3 = pb0 + 3 * 16;
    const int8_t* pb4 = pb0 + 4 * 16;
    const int8_t* pb5 = pb0 + 5 * 16;
    const int8_t* pb6 = pb0 + 6 * 16;
    const int8_t* pb7 = pb0 + 7 * 16;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 4;
    size_t k_leftover = k - (nk << 4);


    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256();
    __m256i c3 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        const __m256i b0 = load_int8_to_int16(pb0);
        const __m256i b1 = load_int8_to_int16(pb1);
        const __m256i b2 = load_int8_to_int16(pb2);
        const __m256i b3 = load_int8_to_int16(pb3);
        const __m256i b4 = load_int8_to_int16(pb4);
        const __m256i b5 = load_int8_to_int16(pb5);
        const __m256i b6 = load_int8_to_int16(pb6);
        const __m256i b7 = load_int8_to_int16(pb7);
#pragma unroll

        for (int m_index = 0; m_index < m; m_index++) {
            if (k == 0) {
                __m256i c0 = _mm256_setzero_si256();
                const __v8si a0 = (__v8si)load_int8_to_int16(a + m_index * lda + k * 16);
                load_2int16_madd(a0[0], b0, c0);
                load_2int16_madd(a0[1], b1, c0);
                load_2int16_madd(a0[2], b2, c0);
                load_2int16_madd(a0[3], b3, c0);
                load_2int16_madd(a0[4], b4, c0);
                load_2int16_madd(a0[5], b5, c0);
                load_2int16_madd(a0[6], b6, c0);
                load_2int16_madd(a0[7], b7, c0);
                _mm256_storeu_si256((__m256i*)(c + m_index * ldc), c0);
            } else {
                __m256i c0 = _mm256_loadu_si256((__m256i*)(c + m_index * ldc));
                const __v8si a0 = (__v8si)load_int8_to_int16(a + m_index * lda + k * 16);
                load_2int16_madd(a0[0], b0, c0);
                load_2int16_madd(a0[1], b1, c0);
                load_2int16_madd(a0[2], b2, c0);
                load_2int16_madd(a0[3], b3, c0);
                load_2int16_madd(a0[4], b4, c0);
                load_2int16_madd(a0[5], b5, c0);
                load_2int16_madd(a0[6], b6, c0);
                load_2int16_madd(a0[7], b7, c0);
                _mm256_storeu_si256((__m256i*)(c + m_index * ldc), c0);
            }
        }

        pb0 += 16 * 8;
        pb1 += 16 * 8;
        pb2 += 16 * 8;
        pb3 += 16 * 8;
        pb4 += 16 * 8;
        pb5 += 16 * 8;
        pb6 += 16 * 8;
        pb7 += 16 * 8;

    }
}
void block4x8_kernel_avx2_me(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {

    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * 16;
    const int8_t* pb2 = pb0 + 2 * 16;
    const int8_t* pb3 = pb0 + 3 * 16;
    const int8_t* pb4 = pb0 + 4 * 16;
    const int8_t* pb5 = pb0 + 5 * 16;
    const int8_t* pb6 = pb0 + 6 * 16;
    const int8_t* pb7 = pb0 + 7 * 16;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 4;
    size_t k_leftover = k - (nk << 4);


    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256();
    __m256i c3 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        const __m256i b0 = load_int8_to_int16(pb0);
        const __m256i b1 = load_int8_to_int16(pb1);
        const __m256i b2 = load_int8_to_int16(pb2);
        const __m256i b3 = load_int8_to_int16(pb3);
        const __m256i b4 = load_int8_to_int16(pb4);
        const __m256i b5 = load_int8_to_int16(pb5);
        const __m256i b6 = load_int8_to_int16(pb6);
        const __m256i b7 = load_int8_to_int16(pb7);

        const __v8si a0 = (__v8si)load_int8_to_int16(pa0);
        const __v8si a1 = (__v8si)load_int8_to_int16(pa1);
        const __v8si a2 = (__v8si)load_int8_to_int16(pa2);
        const __v8si a3 = (__v8si)load_int8_to_int16(pa3);

        load_2int16_madd(a0[0], b0, c0);
        load_2int16_madd(a0[1], b1, c0);
        load_2int16_madd(a0[2], b2, c0);
        load_2int16_madd(a0[3], b3, c0);
        load_2int16_madd(a0[4], b4, c0);
        load_2int16_madd(a0[5], b5, c0);
        load_2int16_madd(a0[6], b6, c0);
        load_2int16_madd(a0[7], b7, c0);

        load_2int16_madd(a1[0], b0, c1);
        load_2int16_madd(a1[1], b1, c1);
        load_2int16_madd(a1[2], b2, c1);
        load_2int16_madd(a1[3], b3, c1);
        load_2int16_madd(a1[4], b4, c1);
        load_2int16_madd(a1[5], b5, c1);
        load_2int16_madd(a1[6], b6, c1);
        load_2int16_madd(a1[7], b7, c1);

        load_2int16_madd(a2[0], b0, c2);
        load_2int16_madd(a2[1], b1, c2);
        load_2int16_madd(a2[2], b2, c2);
        load_2int16_madd(a2[3], b3, c2);
        load_2int16_madd(a2[4], b4, c2);
        load_2int16_madd(a2[5], b5, c2);
        load_2int16_madd(a2[6], b6, c2);
        load_2int16_madd(a2[7], b7, c2);

        load_2int16_madd(a3[0], b0, c3);
        load_2int16_madd(a3[1], b1, c3);
        load_2int16_madd(a3[2], b2, c3);
        load_2int16_madd(a3[3], b3, c3);
        load_2int16_madd(a3[4], b4, c3);
        load_2int16_madd(a3[5], b5, c3);
        load_2int16_madd(a3[6], b6, c3);
        load_2int16_madd(a3[7], b7, c3);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;

        pb0 += 16 * 8;
        pb1 += 16 * 8;
        pb2 += 16 * 8;
        pb3 += 16 * 8;
        pb4 += 16 * 8;
        pb5 += 16 * 8;
        pb6 += 16 * 8;
        pb7 += 16 * 8;

    }

    _mm256_storeu_si256((__m256i*)pc0, c0);
    _mm256_storeu_si256((__m256i*)pc1, c1);
    _mm256_storeu_si256((__m256i*)pc2, c2);
    _mm256_storeu_si256((__m256i*)pc3, c3);
}

void block4x8_kernel_avx2_k2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {

    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * 16;
    const int8_t* pb2 = pb0 + 2 * 16;
    const int8_t* pb3 = pb0 + 3 * 16;
    const int8_t* pb4 = pb0 + 4 * 16;
    const int8_t* pb5 = pb0 + 5 * 16;
    const int8_t* pb6 = pb0 + 6 * 16;
    const int8_t* pb7 = pb0 + 7 * 16;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 4;
    size_t k_leftover = k - (nk << 4);


    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256();
    __m256i c3 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        const __m256i b0 = load_int8_to_int16(pb0);
        const __m256i b1 = load_int8_to_int16(pb1);
        const __m256i b2 = load_int8_to_int16(pb2);
        const __m256i b3 = load_int8_to_int16(pb3);
        const __m256i b4 = load_int8_to_int16(pb4);
        const __m256i b5 = load_int8_to_int16(pb5);
        const __m256i b6 = load_int8_to_int16(pb6);
        const __m256i b7 = load_int8_to_int16(pb7);

        const __v8si a0 = (__v8si)load_int8_to_int16(pa0);
        const __v8si a1 = (__v8si)load_int8_to_int16(pa1);
        const __v8si a2 = (__v8si)load_int8_to_int16(pa2);
        const __v8si a3 = (__v8si)load_int8_to_int16(pa3);



        load_2int16_madd(a0[0], b0, c0);
        load_2int16_madd(a0[1], b1, c0);
        load_2int16_madd(a0[2], b2, c0);
        load_2int16_madd(a0[3], b3, c0);
        load_2int16_madd(a0[4], b4, c0);
        load_2int16_madd(a0[5], b5, c0);
        load_2int16_madd(a0[6], b6, c0);
        load_2int16_madd(a0[7], b7, c0);

        load_2int16_madd(a1[0], b0, c1);
        load_2int16_madd(a1[1], b1, c1);
        load_2int16_madd(a1[2], b2, c1);
        load_2int16_madd(a1[3], b3, c1);
        load_2int16_madd(a1[4], b4, c1);
        load_2int16_madd(a1[5], b5, c1);
        load_2int16_madd(a1[6], b6, c1);
        load_2int16_madd(a1[7], b7, c1);

        load_2int16_madd(a2[0], b0, c2);
        load_2int16_madd(a2[1], b1, c2);
        load_2int16_madd(a2[2], b2, c2);
        load_2int16_madd(a2[3], b3, c2);
        load_2int16_madd(a2[4], b4, c2);
        load_2int16_madd(a2[5], b5, c2);
        load_2int16_madd(a2[6], b6, c2);
        load_2int16_madd(a2[7], b7, c2);

        load_2int16_madd(a3[0], b0, c3);
        load_2int16_madd(a3[1], b1, c3);
        load_2int16_madd(a3[2], b2, c3);
        load_2int16_madd(a3[3], b3, c3);
        load_2int16_madd(a3[4], b4, c3);
        load_2int16_madd(a3[5], b5, c3);
        load_2int16_madd(a3[6], b6, c3);
        load_2int16_madd(a3[7], b7, c3);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;

        pb0 += 16 * 8;
        pb1 += 16 * 8;
        pb2 += 16 * 8;
        pb3 += 16 * 8;
        pb4 += 16 * 8;
        pb5 += 16 * 8;
        pb6 += 16 * 8;
        pb7 += 16 * 8;

    }

    _mm256_storeu_si256((__m256i*)pc0, c0);
    _mm256_storeu_si256((__m256i*)pc1, c1);
    _mm256_storeu_si256((__m256i*)pc2, c2);
    _mm256_storeu_si256((__m256i*)pc3, c3);
}

void block4x64_kernel_avx2_split_k(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;
    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * 16;
    const int8_t* pb2 = pb0 + 2 * 16;
    const int8_t* pb3 = pb0 + 3 * 16;
    const int8_t* pb4 = pb0 + 4 * 16;
    const int8_t* pb5 = pb0 + 5 * 16;
    const int8_t* pb6 = pb0 + 6 * 16;
    const int8_t* pb7 = pb0 + 7 * 16;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;


    size_t nk = k >> 4;
    size_t k_leftover = k - (nk << 4);

    __m256i c0_0 = _mm256_setzero_si256();
    __m256i c0_1 = _mm256_setzero_si256();
    __m256i c0_2 = _mm256_setzero_si256();
    __m256i c0_3 = _mm256_setzero_si256();
    __m256i c0_4 = _mm256_setzero_si256();
    __m256i c0_5 = _mm256_setzero_si256();
    __m256i c0_6 = _mm256_setzero_si256();
    __m256i c0_7 = _mm256_setzero_si256();
    __m256i c1_0 = _mm256_setzero_si256();
    __m256i c1_1 = _mm256_setzero_si256();
    __m256i c1_2 = _mm256_setzero_si256();
    __m256i c1_3 = _mm256_setzero_si256();
    __m256i c1_4 = _mm256_setzero_si256();
    __m256i c1_5 = _mm256_setzero_si256();
    __m256i c1_6 = _mm256_setzero_si256();
    __m256i c1_7 = _mm256_setzero_si256();
    __m256i c2_0 = _mm256_setzero_si256();
    __m256i c2_1 = _mm256_setzero_si256();
    __m256i c2_2 = _mm256_setzero_si256();
    __m256i c2_3 = _mm256_setzero_si256();
    __m256i c2_4 = _mm256_setzero_si256();
    __m256i c2_5 = _mm256_setzero_si256();
    __m256i c2_6 = _mm256_setzero_si256();
    __m256i c2_7 = _mm256_setzero_si256();
    __m256i c3_0 = _mm256_setzero_si256();
    __m256i c3_1 = _mm256_setzero_si256();
    __m256i c3_2 = _mm256_setzero_si256();
    __m256i c3_3 = _mm256_setzero_si256();
    __m256i c3_4 = _mm256_setzero_si256();
    __m256i c3_5 = _mm256_setzero_si256();
    __m256i c3_6 = _mm256_setzero_si256();
    __m256i c3_7 = _mm256_setzero_si256();


    for (size_t k = 0; k < nk; ++k) {

        __v8si a0 = (__v8si)load_int8_to_int16(pa0);
        __v8si a1 = (__v8si)load_int8_to_int16(pa1);
        __v8si a2 = (__v8si)load_int8_to_int16(pa2);
        __v8si a3 = (__v8si)load_int8_to_int16(pa3);

        //        short* a0=(short*)pa0;
        //        short* a1=(short*)pa1;
        //        short* a2=(short*)pa2;
        //        short* a3=(short*)pa3;

        __m256i b0 = load_int8_to_int16(pb0);
        __m256i b1 = load_int8_to_int16(pb1);
        __m256i b2 = load_int8_to_int16(pb2);
        __m256i b3 = load_int8_to_int16(pb3);
        __m256i b4 = load_int8_to_int16(pb4);
        __m256i b5 = load_int8_to_int16(pb5);
        __m256i b6 = load_int8_to_int16(pb6);
        __m256i b7 = load_int8_to_int16(pb7);

        load_2int16_madd(a0[0], b0, c0_0);
        load_2int16_madd(a0[0], b1, c0_1);
        load_2int16_madd(a0[0], b2, c0_2);
        load_2int16_madd(a0[0], b3, c0_3);
        load_2int16_madd(a0[0], b4, c0_4);
        load_2int16_madd(a0[0], b5, c0_5);
        load_2int16_madd(a0[0], b6, c0_6);
        load_2int16_madd(a0[0], b7, c0_7);

        load_2int16_madd(a1[0], b0, c1_0);
        load_2int16_madd(a1[0], b1, c1_1);
        load_2int16_madd(a1[0], b2, c1_2);
        load_2int16_madd(a1[0], b3, c1_3);
        load_2int16_madd(a1[0], b4, c1_4);
        load_2int16_madd(a1[0], b5, c1_5);
        load_2int16_madd(a1[0], b6, c1_6);
        load_2int16_madd(a1[0], b7, c1_7);

        load_2int16_madd(a2[0], b0, c2_0);
        load_2int16_madd(a2[0], b1, c2_1);
        load_2int16_madd(a2[0], b2, c2_2);
        load_2int16_madd(a2[0], b3, c2_3);
        load_2int16_madd(a2[0], b4, c2_4);
        load_2int16_madd(a2[0], b5, c2_5);
        load_2int16_madd(a2[0], b6, c2_6);
        load_2int16_madd(a2[0], b7, c2_7);

        load_2int16_madd(a3[0], b0, c3_0);
        load_2int16_madd(a3[0], b1, c3_1);
        load_2int16_madd(a3[0], b2, c3_2);
        load_2int16_madd(a3[0], b3, c3_3);
        load_2int16_madd(a3[0], b4, c3_4);
        load_2int16_madd(a3[0], b5, c3_5);
        load_2int16_madd(a3[0], b6, c3_6);
        load_2int16_madd(a3[0], b7, c3_7);

        pb0 += 8 * 16;
        pb1 += 8 * 16;
        pb2 += 8 * 16;
        pb3 += 8 * 16;
        pb4 += 8 * 16;
        pb5 += 8 * 16;
        pb6 += 8 * 16;
        pb7 += 8 * 16;

        b0 = load_int8_to_int16(pb0);
        b1 = load_int8_to_int16(pb1);
        b2 = load_int8_to_int16(pb2);
        b3 = load_int8_to_int16(pb3);
        b4 = load_int8_to_int16(pb4);
        b5 = load_int8_to_int16(pb5);
        b6 = load_int8_to_int16(pb6);
        b7 = load_int8_to_int16(pb7);

        load_2int16_madd(a0[1], b0, c0_0);
        load_2int16_madd(a0[1], b1, c0_1);
        load_2int16_madd(a0[1], b2, c0_2);
        load_2int16_madd(a0[1], b3, c0_3);
        load_2int16_madd(a0[1], b4, c0_4);
        load_2int16_madd(a0[1], b5, c0_5);
        load_2int16_madd(a0[1], b6, c0_6);
        load_2int16_madd(a0[1], b7, c0_7);

        load_2int16_madd(a1[1], b0, c1_0);
        load_2int16_madd(a1[1], b1, c1_1);
        load_2int16_madd(a1[1], b2, c1_2);
        load_2int16_madd(a1[1], b3, c1_3);
        load_2int16_madd(a1[1], b4, c1_4);
        load_2int16_madd(a1[1], b5, c1_5);
        load_2int16_madd(a1[1], b6, c1_6);
        load_2int16_madd(a1[1], b7, c1_7);

        load_2int16_madd(a2[1], b0, c2_0);
        load_2int16_madd(a2[1], b1, c2_1);
        load_2int16_madd(a2[1], b2, c2_2);
        load_2int16_madd(a2[1], b3, c2_3);
        load_2int16_madd(a2[1], b4, c2_4);
        load_2int16_madd(a2[1], b5, c2_5);
        load_2int16_madd(a2[1], b6, c2_6);
        load_2int16_madd(a2[1], b7, c2_7);

        load_2int16_madd(a3[1], b0, c3_0);
        load_2int16_madd(a3[1], b1, c3_1);
        load_2int16_madd(a3[1], b2, c3_2);
        load_2int16_madd(a3[1], b3, c3_3);
        load_2int16_madd(a3[1], b4, c3_4);
        load_2int16_madd(a3[1], b5, c3_5);
        load_2int16_madd(a3[1], b6, c3_6);
        load_2int16_madd(a3[1], b7, c3_7);

        pb0 += 8 * 16;
        pb1 += 8 * 16;
        pb2 += 8 * 16;
        pb3 += 8 * 16;
        pb4 += 8 * 16;
        pb5 += 8 * 16;
        pb6 += 8 * 16;
        pb7 += 8 * 16;

        b0 = load_int8_to_int16(pb0);
        b1 = load_int8_to_int16(pb1);
        b2 = load_int8_to_int16(pb2);
        b3 = load_int8_to_int16(pb3);
        b4 = load_int8_to_int16(pb4);
        b5 = load_int8_to_int16(pb5);
        b6 = load_int8_to_int16(pb6);
        b7 = load_int8_to_int16(pb7);

        load_2int16_madd(a0[2], b0, c0_0);
        load_2int16_madd(a0[2], b1, c0_1);
        load_2int16_madd(a0[2], b2, c0_2);
        load_2int16_madd(a0[2], b3, c0_3);
        load_2int16_madd(a0[2], b4, c0_4);
        load_2int16_madd(a0[2], b5, c0_5);
        load_2int16_madd(a0[2], b6, c0_6);
        load_2int16_madd(a0[2], b7, c0_7);

        load_2int16_madd(a1[2], b0, c1_0);
        load_2int16_madd(a1[2], b1, c1_1);
        load_2int16_madd(a1[2], b2, c1_2);
        load_2int16_madd(a1[2], b3, c1_3);
        load_2int16_madd(a1[2], b4, c1_4);
        load_2int16_madd(a1[2], b5, c1_5);
        load_2int16_madd(a1[2], b6, c1_6);
        load_2int16_madd(a1[2], b7, c1_7);

        load_2int16_madd(a2[2], b0, c2_0);
        load_2int16_madd(a2[2], b1, c2_1);
        load_2int16_madd(a2[2], b2, c2_2);
        load_2int16_madd(a2[2], b3, c2_3);
        load_2int16_madd(a2[2], b4, c2_4);
        load_2int16_madd(a2[2], b5, c2_5);
        load_2int16_madd(a2[2], b6, c2_6);
        load_2int16_madd(a2[2], b7, c2_7);

        load_2int16_madd(a3[2], b0, c3_0);
        load_2int16_madd(a3[2], b1, c3_1);
        load_2int16_madd(a3[2], b2, c3_2);
        load_2int16_madd(a3[2], b3, c3_3);
        load_2int16_madd(a3[2], b4, c3_4);
        load_2int16_madd(a3[2], b5, c3_5);
        load_2int16_madd(a3[2], b6, c3_6);
        load_2int16_madd(a3[2], b7, c3_7);

        pb0 += 8 * 16;
        pb1 += 8 * 16;
        pb2 += 8 * 16;
        pb3 += 8 * 16;
        pb4 += 8 * 16;
        pb5 += 8 * 16;
        pb6 += 8 * 16;
        pb7 += 8 * 16;

        b0 = load_int8_to_int16(pb0);
        b1 = load_int8_to_int16(pb1);
        b2 = load_int8_to_int16(pb2);
        b3 = load_int8_to_int16(pb3);
        b4 = load_int8_to_int16(pb4);
        b5 = load_int8_to_int16(pb5);
        b6 = load_int8_to_int16(pb6);
        b7 = load_int8_to_int16(pb7);

        load_2int16_madd(a0[3], b0, c0_0);
        load_2int16_madd(a0[3], b1, c0_1);
        load_2int16_madd(a0[3], b2, c0_2);
        load_2int16_madd(a0[3], b3, c0_3);
        load_2int16_madd(a0[3], b4, c0_4);
        load_2int16_madd(a0[3], b5, c0_5);
        load_2int16_madd(a0[3], b6, c0_6);
        load_2int16_madd(a0[3], b7, c0_7);

        load_2int16_madd(a1[3], b0, c1_0);
        load_2int16_madd(a1[3], b1, c1_1);
        load_2int16_madd(a1[3], b2, c1_2);
        load_2int16_madd(a1[3], b3, c1_3);
        load_2int16_madd(a1[3], b4, c1_4);
        load_2int16_madd(a1[3], b5, c1_5);
        load_2int16_madd(a1[3], b6, c1_6);
        load_2int16_madd(a1[3], b7, c1_7);

        load_2int16_madd(a2[3], b0, c2_0);
        load_2int16_madd(a2[3], b1, c2_1);
        load_2int16_madd(a2[3], b2, c2_2);
        load_2int16_madd(a2[3], b3, c2_3);
        load_2int16_madd(a2[3], b4, c2_4);
        load_2int16_madd(a2[3], b5, c2_5);
        load_2int16_madd(a2[3], b6, c2_6);
        load_2int16_madd(a2[3], b7, c2_7);

        load_2int16_madd(a3[3], b0, c3_0);
        load_2int16_madd(a3[3], b1, c3_1);
        load_2int16_madd(a3[3], b2, c3_2);
        load_2int16_madd(a3[3], b3, c3_3);
        load_2int16_madd(a3[3], b4, c3_4);
        load_2int16_madd(a3[3], b5, c3_5);
        load_2int16_madd(a3[3], b6, c3_6);
        load_2int16_madd(a3[3], b7, c3_7);

        pb0 += 8 * 16;
        pb1 += 8 * 16;
        pb2 += 8 * 16;
        pb3 += 8 * 16;
        pb4 += 8 * 16;
        pb5 += 8 * 16;
        pb6 += 8 * 16;
        pb7 += 8 * 16;

        b0 = load_int8_to_int16(pb0);
        b1 = load_int8_to_int16(pb1);
        b2 = load_int8_to_int16(pb2);
        b3 = load_int8_to_int16(pb3);
        b4 = load_int8_to_int16(pb4);
        b5 = load_int8_to_int16(pb5);
        b6 = load_int8_to_int16(pb6);
        b7 = load_int8_to_int16(pb7);

        load_2int16_madd(a0[4], b0, c0_0);
        load_2int16_madd(a0[4], b1, c0_1);
        load_2int16_madd(a0[4], b2, c0_2);
        load_2int16_madd(a0[4], b3, c0_3);
        load_2int16_madd(a0[4], b4, c0_4);
        load_2int16_madd(a0[4], b5, c0_5);
        load_2int16_madd(a0[4], b6, c0_6);
        load_2int16_madd(a0[4], b7, c0_7);

        load_2int16_madd(a1[4], b0, c1_0);
        load_2int16_madd(a1[4], b1, c1_1);
        load_2int16_madd(a1[4], b2, c1_2);
        load_2int16_madd(a1[4], b3, c1_3);
        load_2int16_madd(a1[4], b4, c1_4);
        load_2int16_madd(a1[4], b5, c1_5);
        load_2int16_madd(a1[4], b6, c1_6);
        load_2int16_madd(a1[4], b7, c1_7);

        load_2int16_madd(a2[4], b0, c2_0);
        load_2int16_madd(a2[4], b1, c2_1);
        load_2int16_madd(a2[4], b2, c2_2);
        load_2int16_madd(a2[4], b3, c2_3);
        load_2int16_madd(a2[4], b4, c2_4);
        load_2int16_madd(a2[4], b5, c2_5);
        load_2int16_madd(a2[4], b6, c2_6);
        load_2int16_madd(a2[4], b7, c2_7);

        load_2int16_madd(a3[4], b0, c3_0);
        load_2int16_madd(a3[4], b1, c3_1);
        load_2int16_madd(a3[4], b2, c3_2);
        load_2int16_madd(a3[4], b3, c3_3);
        load_2int16_madd(a3[4], b4, c3_4);
        load_2int16_madd(a3[4], b5, c3_5);
        load_2int16_madd(a3[4], b6, c3_6);
        load_2int16_madd(a3[4], b7, c3_7);

        pb0 += 8 * 16;
        pb1 += 8 * 16;
        pb2 += 8 * 16;
        pb3 += 8 * 16;
        pb4 += 8 * 16;
        pb5 += 8 * 16;
        pb6 += 8 * 16;
        pb7 += 8 * 16;

        b0 = load_int8_to_int16(pb0);
        b1 = load_int8_to_int16(pb1);
        b2 = load_int8_to_int16(pb2);
        b3 = load_int8_to_int16(pb3);
        b4 = load_int8_to_int16(pb4);
        b5 = load_int8_to_int16(pb5);
        b6 = load_int8_to_int16(pb6);
        b7 = load_int8_to_int16(pb7);

        load_2int16_madd(a0[5], b0, c0_0);
        load_2int16_madd(a0[5], b1, c0_1);
        load_2int16_madd(a0[5], b2, c0_2);
        load_2int16_madd(a0[5], b3, c0_3);
        load_2int16_madd(a0[5], b4, c0_4);
        load_2int16_madd(a0[5], b5, c0_5);
        load_2int16_madd(a0[5], b6, c0_6);
        load_2int16_madd(a0[5], b7, c0_7);

        load_2int16_madd(a1[5], b0, c1_0);
        load_2int16_madd(a1[5], b1, c1_1);
        load_2int16_madd(a1[5], b2, c1_2);
        load_2int16_madd(a1[5], b3, c1_3);
        load_2int16_madd(a1[5], b4, c1_4);
        load_2int16_madd(a1[5], b5, c1_5);
        load_2int16_madd(a1[5], b6, c1_6);
        load_2int16_madd(a1[5], b7, c1_7);

        load_2int16_madd(a2[5], b0, c2_0);
        load_2int16_madd(a2[5], b1, c2_1);
        load_2int16_madd(a2[5], b2, c2_2);
        load_2int16_madd(a2[5], b3, c2_3);
        load_2int16_madd(a2[5], b4, c2_4);
        load_2int16_madd(a2[5], b5, c2_5);
        load_2int16_madd(a2[5], b6, c2_6);
        load_2int16_madd(a2[5], b7, c2_7);

        load_2int16_madd(a3[5], b0, c3_0);
        load_2int16_madd(a3[5], b1, c3_1);
        load_2int16_madd(a3[5], b2, c3_2);
        load_2int16_madd(a3[5], b3, c3_3);
        load_2int16_madd(a3[5], b4, c3_4);
        load_2int16_madd(a3[5], b5, c3_5);
        load_2int16_madd(a3[5], b6, c3_6);
        load_2int16_madd(a3[5], b7, c3_7);

        pb0 += 8 * 16;
        pb1 += 8 * 16;
        pb2 += 8 * 16;
        pb3 += 8 * 16;
        pb4 += 8 * 16;
        pb5 += 8 * 16;
        pb6 += 8 * 16;
        pb7 += 8 * 16;

        b0 = load_int8_to_int16(pb0);
        b1 = load_int8_to_int16(pb1);
        b2 = load_int8_to_int16(pb2);
        b3 = load_int8_to_int16(pb3);
        b4 = load_int8_to_int16(pb4);
        b5 = load_int8_to_int16(pb5);
        b6 = load_int8_to_int16(pb6);
        b7 = load_int8_to_int16(pb7);

        load_2int16_madd(a0[6], b0, c0_0);
        load_2int16_madd(a0[6], b1, c0_1);
        load_2int16_madd(a0[6], b2, c0_2);
        load_2int16_madd(a0[6], b3, c0_3);
        load_2int16_madd(a0[6], b4, c0_4);
        load_2int16_madd(a0[6], b5, c0_5);
        load_2int16_madd(a0[6], b6, c0_6);
        load_2int16_madd(a0[6], b7, c0_7);

        load_2int16_madd(a1[6], b0, c1_0);
        load_2int16_madd(a1[6], b1, c1_1);
        load_2int16_madd(a1[6], b2, c1_2);
        load_2int16_madd(a1[6], b3, c1_3);
        load_2int16_madd(a1[6], b4, c1_4);
        load_2int16_madd(a1[6], b5, c1_5);
        load_2int16_madd(a1[6], b6, c1_6);
        load_2int16_madd(a1[6], b7, c1_7);

        load_2int16_madd(a2[6], b0, c2_0);
        load_2int16_madd(a2[6], b1, c2_1);
        load_2int16_madd(a2[6], b2, c2_2);
        load_2int16_madd(a2[6], b3, c2_3);
        load_2int16_madd(a2[6], b4, c2_4);
        load_2int16_madd(a2[6], b5, c2_5);
        load_2int16_madd(a2[6], b6, c2_6);
        load_2int16_madd(a2[6], b7, c2_7);

        load_2int16_madd(a3[6], b0, c3_0);
        load_2int16_madd(a3[6], b1, c3_1);
        load_2int16_madd(a3[6], b2, c3_2);
        load_2int16_madd(a3[6], b3, c3_3);
        load_2int16_madd(a3[6], b4, c3_4);
        load_2int16_madd(a3[6], b5, c3_5);
        load_2int16_madd(a3[6], b6, c3_6);
        load_2int16_madd(a3[6], b7, c3_7);

        pb0 += 8 * 16;
        pb1 += 8 * 16;
        pb2 += 8 * 16;
        pb3 += 8 * 16;
        pb4 += 8 * 16;
        pb5 += 8 * 16;
        pb6 += 8 * 16;
        pb7 += 8 * 16;

        b0 = load_int8_to_int16(pb0);
        b1 = load_int8_to_int16(pb1);
        b2 = load_int8_to_int16(pb2);
        b3 = load_int8_to_int16(pb3);
        b4 = load_int8_to_int16(pb4);
        b5 = load_int8_to_int16(pb5);
        b6 = load_int8_to_int16(pb6);
        b7 = load_int8_to_int16(pb7);

        load_2int16_madd(a0[7], b0, c0_0);
        load_2int16_madd(a0[7], b1, c0_1);
        load_2int16_madd(a0[7], b2, c0_2);
        load_2int16_madd(a0[7], b3, c0_3);
        load_2int16_madd(a0[7], b4, c0_4);
        load_2int16_madd(a0[7], b5, c0_5);
        load_2int16_madd(a0[7], b6, c0_6);
        load_2int16_madd(a0[7], b7, c0_7);

        load_2int16_madd(a1[7], b0, c1_0);
        load_2int16_madd(a1[7], b1, c1_1);
        load_2int16_madd(a1[7], b2, c1_2);
        load_2int16_madd(a1[7], b3, c1_3);
        load_2int16_madd(a1[7], b4, c1_4);
        load_2int16_madd(a1[7], b5, c1_5);
        load_2int16_madd(a1[7], b6, c1_6);
        load_2int16_madd(a1[7], b7, c1_7);

        load_2int16_madd(a2[7], b0, c2_0);
        load_2int16_madd(a2[7], b1, c2_1);
        load_2int16_madd(a2[7], b2, c2_2);
        load_2int16_madd(a2[7], b3, c2_3);
        load_2int16_madd(a2[7], b4, c2_4);
        load_2int16_madd(a2[7], b5, c2_5);
        load_2int16_madd(a2[7], b6, c2_6);
        load_2int16_madd(a2[7], b7, c2_7);

        load_2int16_madd(a3[7], b0, c3_0);
        load_2int16_madd(a3[7], b1, c3_1);
        load_2int16_madd(a3[7], b2, c3_2);
        load_2int16_madd(a3[7], b3, c3_3);
        load_2int16_madd(a3[7], b4, c3_4);
        load_2int16_madd(a3[7], b5, c3_5);
        load_2int16_madd(a3[7], b6, c3_6);
        load_2int16_madd(a3[7], b7, c3_7);

        pb0 += 8 * 16;
        pb1 += 8 * 16;
        pb2 += 8 * 16;
        pb3 += 8 * 16;
        pb4 += 8 * 16;
        pb5 += 8 * 16;
        pb6 += 8 * 16;
        pb7 += 8 * 16;

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;
    }

    _mm256_storeu_si256((__m256i*)(pc0 + 0 * 8), c0_0);
    _mm256_storeu_si256((__m256i*)(pc0 + 1 * 8), c0_1);
    _mm256_storeu_si256((__m256i*)(pc0 + 2 * 8), c0_2);
    _mm256_storeu_si256((__m256i*)(pc0 + 3 * 8), c0_3);
    _mm256_storeu_si256((__m256i*)(pc0 + 4 * 8), c0_4);
    _mm256_storeu_si256((__m256i*)(pc0 + 5 * 8), c0_5);
    _mm256_storeu_si256((__m256i*)(pc0 + 6 * 8), c0_6);
    _mm256_storeu_si256((__m256i*)(pc0 + 7 * 8), c0_7);

    _mm256_storeu_si256((__m256i*)(pc1 + 0 * 8), c1_0);
    _mm256_storeu_si256((__m256i*)(pc1 + 1 * 8), c1_1);
    _mm256_storeu_si256((__m256i*)(pc1 + 2 * 8), c1_2);
    _mm256_storeu_si256((__m256i*)(pc1 + 3 * 8), c1_3);
    _mm256_storeu_si256((__m256i*)(pc1 + 4 * 8), c1_4);
    _mm256_storeu_si256((__m256i*)(pc1 + 5 * 8), c1_5);
    _mm256_storeu_si256((__m256i*)(pc1 + 6 * 8), c1_6);
    _mm256_storeu_si256((__m256i*)(pc1 + 7 * 8), c1_7);

    _mm256_storeu_si256((__m256i*)(pc2 + 0 * 8), c2_0);
    _mm256_storeu_si256((__m256i*)(pc2 + 1 * 8), c2_1);
    _mm256_storeu_si256((__m256i*)(pc2 + 2 * 8), c2_2);
    _mm256_storeu_si256((__m256i*)(pc2 + 3 * 8), c2_3);
    _mm256_storeu_si256((__m256i*)(pc2 + 4 * 8), c2_4);
    _mm256_storeu_si256((__m256i*)(pc2 + 5 * 8), c2_5);
    _mm256_storeu_si256((__m256i*)(pc2 + 6 * 8), c2_6);
    _mm256_storeu_si256((__m256i*)(pc2 + 7 * 8), c2_7);

    _mm256_storeu_si256((__m256i*)(pc3 + 0 * 8), c3_0);
    _mm256_storeu_si256((__m256i*)(pc3 + 1 * 8), c3_1);
    _mm256_storeu_si256((__m256i*)(pc3 + 2 * 8), c3_2);
    _mm256_storeu_si256((__m256i*)(pc3 + 3 * 8), c3_3);
    _mm256_storeu_si256((__m256i*)(pc3 + 4 * 8), c3_4);
    _mm256_storeu_si256((__m256i*)(pc3 + 5 * 8), c3_5);
    _mm256_storeu_si256((__m256i*)(pc3 + 6 * 8), c3_6);
    _mm256_storeu_si256((__m256i*)(pc3 + 7 * 8), c3_7);

}

inline void avx_s8s8s32_gemm_4x8_packed_dot_add(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    const int m_block = 4;
    const int n_block = 8;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0";

#if USE_OMP_IN_INTRINSIC_PACKED_FC
    #pragma omp parallel for schedule(static)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];

            int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block4x8_kernel_avx2_me(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
        }
    }

}

inline void avx_s8s8s32_gemm_4x64_packed_split_k(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    const int m_block = 4;
    const int n_block = 64;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0";

#if USE_OMP_IN_INTRINSIC_PACKED_FC
    #pragma omp parallel for schedule(static)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];

            int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block4x64_kernel_avx2_split_k(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
        }
    }

}

inline void avx_s8s8s32_gemm_mx8_packed_dot_add(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    const int m_block = 4;
    const int n_block = 8;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0";

#if USE_OMP_IN_INTRINSIC_PACKED_FC
    #pragma omp parallel for schedule(static)
#endif

    for (int nbi = 0; nbi < nb; nbi++) {
        const int8_t* b_ptr = &b[nbi * n_block * ldb];
        int32_t* c_ptr = &c[nbi * n_block];
        block_mx8_kernel_avx2_me(m, k, a, lda, b_ptr, ldb, c_ptr, ldc);
    }
}


void block4x2_kernel_avx2_me(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    //printf("block4x2_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;
    __m256i ma0_h;
    __m256i ma1_h;
    __m256i ma2_h;
    __m256i ma3_h;

    __m256i mb0_l;
    __m256i mb1_l;
    __m256i mb0_h;
    __m256i mb1_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;
    __m256i mc4;
    __m256i mc5;
    __m256i mc6;
    __m256i mc7;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb1 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma0_h, mb1_h));

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);

        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma1_h, mb0_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma1_h, mb1_h));

        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        ma2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa2 + 16)));

        mc4 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb1_l);

        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma2_h, mb0_h));
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma2_h, mb1_h));

        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        ma3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa3 + 16)));

        mc6 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb1_l);

        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma3_h, mb0_h));
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma3_h, mb1_h));

        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;

        pb0 += 32;
        pb1 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));

        mc4 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb1_l);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));

        mc6 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb1_l);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;

        pb0 += 16;
        pb1 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        __m256i ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        __m256i mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));
        __m256i mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa2));

        mc4 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb1_l);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa3));

        mc6 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb1_l);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 8;
        pa1 += 8;
        pa2 += 8;
        pa3 += 8;

        pb0 += 8;
        pb1 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];
            ga2[i] = pa2[i];
            ga3[i] = pa3[i];

            gb0[i] = pb0[i];
            gb1[i] = pb1[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga2));

        mc4 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb1_l);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga3));

        mc6 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb1_l);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);
    }

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1 * stride] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum2 = _mm256_hadd_epi32(sum2, zero);
    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum2, 0);
    pc1[1 * stride] = _mm256_extract_epi32(sum2, 1);

    //the 2 row
    sum4 = _mm256_hadd_epi32(sum4, sum5);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc2[0] = _mm256_extract_epi32(sum4, 0);
    pc2[1 * stride] = _mm256_extract_epi32(sum4, 1);

    //the 3 row
    sum6 = _mm256_hadd_epi32(sum6, sum7);
    sum6 = _mm256_hadd_epi32(sum6, zero);
    sum6 = _mm256_add_epi32(sum6, _mm256_permute2x128_si256(sum6, zero, 0x31));

    pc3[0] = _mm256_extract_epi32(sum6, 0);
    pc3[1 * stride] = _mm256_extract_epi32(sum6, 1);
}

inline void block4x2_kernel_avx2_me_k16(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    //printf("block4x2_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 4; // k / 32
    size_t k_leftover = k - (nk << 4); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;

    __m256i mb0_l;
    __m256i mb1_l;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;
    __m256i mc4;
    __m256i mc5;
    __m256i mc6;
    __m256i mc7;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);


        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);


        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));

        mc4 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb1_l);


        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));

        mc6 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb1_l);


        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;

        pb0 += 16;
        pb1 += 16;
    }

    CHECK_EQ(k_leftover, 0);

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1 * stride] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum2 = _mm256_hadd_epi32(sum2, zero);
    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum2, 0);
    pc1[1 * stride] = _mm256_extract_epi32(sum2, 1);

    //the 2 row
    sum4 = _mm256_hadd_epi32(sum4, sum5);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc2[0] = _mm256_extract_epi32(sum4, 0);
    pc2[1 * stride] = _mm256_extract_epi32(sum4, 1);

    //the 3 row
    sum6 = _mm256_hadd_epi32(sum6, sum7);
    sum6 = _mm256_hadd_epi32(sum6, zero);
    sum6 = _mm256_add_epi32(sum6, _mm256_permute2x128_si256(sum6, zero, 0x31));

    pc3[0] = _mm256_extract_epi32(sum6, 0);
    pc3[1 * stride] = _mm256_extract_epi32(sum6, 1);
}

/**
 * b packed
 * @param k
 * @param a
 * @param lda
 * @param b
 * @param ldb
 * @param c
 * @param ldc
 */
inline void block2x4_kernel_avx2_me_k16(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {
    //printf("block4x2_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;


    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;

    size_t nk = k >> 4; // k / 16
    size_t k_leftover = k - (nk << 4); // k % 16

    __m256i ma0;
    __m256i ma1;

    __m256i mb0;
    __m256i mb1;
    __m256i mb2;
    __m256i mb3;;

    __m256i temp_0;
    __m256i temp_1;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        //b
        mb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        //the 0 row
        temp_0 = _mm256_madd_epi16(ma0, mb0);
        temp_1 = _mm256_madd_epi16(ma1, mb0);
        sum0 = _mm256_add_epi32(sum0, temp_0);
        sum1 = _mm256_add_epi32(sum1, temp_1);


        //the 1 row
        mb1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        temp_0 = _mm256_madd_epi16(ma0, mb1);
        temp_1 = _mm256_madd_epi16(ma1, mb1);

        sum2 = _mm256_add_epi32(sum2, temp_0);
        sum3 = _mm256_add_epi32(sum3, temp_1);


        mb2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        temp_0 = _mm256_madd_epi16(ma0, mb2);
        temp_1 = _mm256_madd_epi16(ma1, mb2);

        sum4 = _mm256_add_epi32(sum4, temp_0);
        sum5 = _mm256_add_epi32(sum5, temp_1);

        //the 3 row

        mb3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        temp_0 = _mm256_madd_epi16(ma0, mb3);
        temp_1 = _mm256_madd_epi16(ma1, mb3);
        sum6 = _mm256_add_epi32(sum6, temp_0);
        sum7 = _mm256_add_epi32(sum7, temp_1);

        pa0 += 16;
        pa1 += 16;

        pb0 += 16;
        pb1 += 16;
        pb2 += 16;
        pb3 += 16;
    }

    CHECK_EQ(k_leftover, 0);

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum2);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum4 = _mm256_hadd_epi32(sum4, sum6);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc0[2] = _mm256_extract_epi32(sum4, 0);
    pc0[3] = _mm256_extract_epi32(sum4, 1);

    //the 2 row
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum1, 0);
    pc1[1] = _mm256_extract_epi32(sum1, 1);

    //the 3 row
    sum5 = _mm256_hadd_epi32(sum5, sum7);
    sum5 = _mm256_hadd_epi32(sum5, zero);
    sum5 = _mm256_add_epi32(sum5, _mm256_permute2x128_si256(sum5, zero, 0x31));

    pc1[2] = _mm256_extract_epi32(sum5, 0);
    pc1[3] = _mm256_extract_epi32(sum5, 1);
}

/**
 * b packed
 * @param k
 * @param a
 * @param lda
 * @param b
 * @param ldb
 * @param c
 * @param ldc
 */
inline void block2x4_kernel_avx2_me_k16_packed(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {
    //printf("block4x2_kernel_avx2\n");

    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;


    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * 16;
    const int8_t* pb2 = pb0 + 2 * 16;
    const int8_t* pb3 = pb0 + 3 * 16;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;

    size_t nk = k >> 4; // k / 16
    size_t k_leftover = k - (nk << 4); // k % 16

    __m256i ma0;
    __m256i ma1;

    __m256i mb0;
    __m256i mb1;
    __m256i mb2;
    __m256i mb3;;

    __m256i temp_0;
    __m256i temp_1;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        //b
        mb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        //the 0 row
        temp_0 = _mm256_madd_epi16(ma0, mb0);
        temp_1 = _mm256_madd_epi16(ma1, mb0);
        sum0 = _mm256_add_epi32(sum0, temp_0);
        sum1 = _mm256_add_epi32(sum1, temp_1);


        //the 1 row
        mb1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        temp_0 = _mm256_madd_epi16(ma0, mb1);
        temp_1 = _mm256_madd_epi16(ma1, mb1);

        sum2 = _mm256_add_epi32(sum2, temp_0);
        sum3 = _mm256_add_epi32(sum3, temp_1);


        mb2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        temp_0 = _mm256_madd_epi16(ma0, mb2);
        temp_1 = _mm256_madd_epi16(ma1, mb2);

        sum4 = _mm256_add_epi32(sum4, temp_0);
        sum5 = _mm256_add_epi32(sum5, temp_1);

        //the 3 row

        mb3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        temp_0 = _mm256_madd_epi16(ma0, mb3);
        temp_1 = _mm256_madd_epi16(ma1, mb3);
        sum6 = _mm256_add_epi32(sum6, temp_0);
        sum7 = _mm256_add_epi32(sum7, temp_1);

        pa0 += 16;
        pa1 += 16;

        pb0 += 16 * 4;
        pb1 += 16 * 4;
        pb2 += 16 * 4;
        pb3 += 16 * 4;
    }

    CHECK_EQ(k_leftover, 0);

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum2);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum4 = _mm256_hadd_epi32(sum4, sum6);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc0[2] = _mm256_extract_epi32(sum4, 0);
    pc0[3] = _mm256_extract_epi32(sum4, 1);

    //the 2 row
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum1, 0);
    pc1[1] = _mm256_extract_epi32(sum1, 1);

    //the 3 row
    sum5 = _mm256_hadd_epi32(sum5, sum7);
    sum5 = _mm256_hadd_epi32(sum5, zero);
    sum5 = _mm256_add_epi32(sum5, _mm256_permute2x128_si256(sum5, zero, 0x31));

    pc1[2] = _mm256_extract_epi32(sum5, 0);
    pc1[3] = _mm256_extract_epi32(sum5, 1);
}


/**
 * b packed
 * @param k
 * @param a
 * @param lda
 * @param b
 * @param ldb
 * @param c
 * @param ldc
 */
inline void block1x8_kernel_avx2_me_k16(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {
    //printf("block4x2_kernel_avx2\n");
    const int8_t* pa0 = a;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;
    const int8_t* pb4 = pb0 + 4 * ldb;
    const int8_t* pb5 = pb0 + 5 * ldb;
    const int8_t* pb6 = pb0 + 6 * ldb;
    const int8_t* pb7 = pb0 + 7 * ldb;

    int* pc0 = c;

    size_t nk = k >> 4; // k / 16
    size_t k_leftover = k - (nk << 4); // k % 16

    __m256i ma0;

    __m256i  mb0;
    __m256i  mb1;
    __m256i  mb2;
    __m256i  mb3;
    __m256i  mb4;
    __m256i  mb5;
    __m256i  mb6;
    __m256i  mb7;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        __m256i temp_0;
        __m256i temp_1;
        __m256i temp_2;
        __m256i temp_3;
        //a
        ma0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        //b
        mb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));

        //the 0 row
        temp_0 = _mm256_madd_epi16(ma0, mb0);
        temp_1 = _mm256_madd_epi16(ma0, mb1);
        sum0 = _mm256_add_epi32(sum0, temp_0);
        sum1 = _mm256_add_epi32(sum1, temp_1);


        //the 1 row
        mb2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        mb3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        temp_2 = _mm256_madd_epi16(ma0, mb2);
        temp_3 = _mm256_madd_epi16(ma0, mb3);

        sum2 = _mm256_add_epi32(sum2, temp_2);
        sum3 = _mm256_add_epi32(sum3, temp_3);


        mb4 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb4));
        mb5 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb5));

        temp_0 = _mm256_madd_epi16(ma0, mb4);
        temp_1 = _mm256_madd_epi16(ma0, mb5);

        sum4 = _mm256_add_epi32(sum4, temp_0);
        sum5 = _mm256_add_epi32(sum5, temp_1);

        //the 3 row

        mb6 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb6));
        mb7 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb7));
        temp_2 = _mm256_madd_epi16(ma0, mb6);
        temp_3 = _mm256_madd_epi16(ma0, mb7);
        sum6 = _mm256_add_epi32(sum6, temp_2);
        sum7 = _mm256_add_epi32(sum7, temp_3);

        pa0 += 16;

        pb0 += 16;
        pb1 += 16;
        pb2 += 16;
        pb3 += 16;
        pb4 += 16;
        pb5 += 16;
        pb6 += 16;
        pb7 += 16;
    }

    CHECK_EQ(k_leftover, 0);

    //store

    //the 0 row
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, sum0, 0x81));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 8));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 4));
    pc0[0] = _mm256_extract_epi32(sum0, 0);

    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, sum1, 0x81));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 8));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 4));
    pc0[1] = _mm256_extract_epi32(sum1, 0);

    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, sum2, 0x81));
    sum2 = _mm256_add_epi32(sum2, _mm256_srli_si256(sum2, 8));
    sum2 = _mm256_add_epi32(sum2, _mm256_srli_si256(sum2, 4));
    pc0[2] = _mm256_extract_epi32(sum2, 0);

    sum3 = _mm256_add_epi32(sum3, _mm256_permute2x128_si256(sum3, sum3, 0x81));
    sum3 = _mm256_add_epi32(sum3, _mm256_srli_si256(sum3, 8));
    sum3 = _mm256_add_epi32(sum3, _mm256_srli_si256(sum3, 4));
    pc0[3] = _mm256_extract_epi32(sum3, 0);

    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, sum4, 0x81));
    sum4 = _mm256_add_epi32(sum4, _mm256_srli_si256(sum4, 8));
    sum4 = _mm256_add_epi32(sum4, _mm256_srli_si256(sum4, 4));
    pc0[4] = _mm256_extract_epi32(sum4, 0);

    sum5 = _mm256_add_epi32(sum5, _mm256_permute2x128_si256(sum5, sum5, 0x81));
    sum5 = _mm256_add_epi32(sum5, _mm256_srli_si256(sum5, 8));
    sum5 = _mm256_add_epi32(sum5, _mm256_srli_si256(sum5, 4));
    pc0[5] = _mm256_extract_epi32(sum5, 0);

    sum6 = _mm256_add_epi32(sum6, _mm256_permute2x128_si256(sum6, sum6, 0x81));
    sum6 = _mm256_add_epi32(sum6, _mm256_srli_si256(sum6, 8));
    sum6 = _mm256_add_epi32(sum6, _mm256_srli_si256(sum6, 4));
    pc0[6] = _mm256_extract_epi32(sum6, 0);

    sum7 = _mm256_add_epi32(sum7, _mm256_permute2x128_si256(sum7, sum7, 0x81));
    sum7 = _mm256_add_epi32(sum7, _mm256_srli_si256(sum7, 8));
    sum7 = _mm256_add_epi32(sum7, _mm256_srli_si256(sum7, 4));
    pc0[7] = _mm256_extract_epi32(sum7, 0);
}


/**
 * b packed
 * @param k
 * @param a
 * @param lda
 * @param b
 * @param ldb
 * @param c
 * @param ldc
 */
inline void block2x4_kernel_avx2_me_k16_pad(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {
    //printf("block4x2_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;


    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;

    size_t nk = k >> 4; // k / 32
    size_t k_leftover = k - (nk << 4); // k % 32

    __m256i ma0;
    __m256i ma1;

    __m256i mb0;
    __m256i mb1;
    __m256i mb2;
    __m256i mb3;;

    __m256i temp_0;
    __m256i temp_1;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        //b
        mb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        //the 0 row
        temp_0 = _mm256_madd_epi16(ma0, mb0);
        temp_1 = _mm256_madd_epi16(ma1, mb0);
        sum0 = _mm256_add_epi32(sum0, temp_0);
        sum1 = _mm256_add_epi32(sum1, temp_1);


        //the 1 row
        mb1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        temp_0 = _mm256_madd_epi16(ma0, mb1);
        temp_1 = _mm256_madd_epi16(ma1, mb1);

        sum2 = _mm256_add_epi32(sum2, temp_0);
        sum3 = _mm256_add_epi32(sum3, temp_1);


        mb2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        temp_0 = _mm256_madd_epi16(ma0, mb2);
        temp_1 = _mm256_madd_epi16(ma1, mb2);

        sum4 = _mm256_add_epi32(sum4, temp_0);
        sum5 = _mm256_add_epi32(sum5, temp_1);

        //the 3 row

        mb3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        temp_0 = _mm256_madd_epi16(ma0, mb3);
        temp_1 = _mm256_madd_epi16(ma1, mb3);
        sum6 = _mm256_add_epi32(sum6, temp_0);
        sum7 = _mm256_add_epi32(sum7, temp_1);

        pa0 += 16;
        pa1 += 16;

        pb0 += 16;
        pb1 += 16;
        pb2 += 16;
        pb3 += 16;
    }

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum2);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum4 = _mm256_hadd_epi32(sum4, sum6);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc0[2] = _mm256_extract_epi32(sum4, 0);
    pc0[3] = _mm256_extract_epi32(sum4, 1);

    //the 2 row
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum1, 0);
    pc1[1] = _mm256_extract_epi32(sum1, 1);

    //the 3 row
    sum5 = _mm256_hadd_epi32(sum5, sum7);
    sum5 = _mm256_hadd_epi32(sum5, zero);
    sum5 = _mm256_add_epi32(sum5, _mm256_permute2x128_si256(sum5, zero, 0x31));

    pc1[2] = _mm256_extract_epi32(sum5, 0);
    pc1[3] = _mm256_extract_epi32(sum5, 1);
}

/**
 * b packed
 * @param k
 * @param a
 * @param lda
 * @param b
 * @param ldb
 * @param c
 * @param ldc
 */
inline void block2x4_kernel_avx2_me_k16_pad_s8s8fp32(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, float* c, const int32_t ldc, const float* scale) {
    //printf("block4x2_kernel_avx2\n");

    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;


    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;

    float* pc0 = c;
    float* pc1 = c + 1 * ldc;

    size_t nk = k >> 4; // k / 32
    size_t k_leftover = k - (nk << 4); // k % 32

    __m256i ma0;
    __m256i ma1;

    __m256i mb0;
    __m256i mb1;
    __m256i mb2;
    __m256i mb3;



    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        __m256i temp_0;
        __m256i temp_1;
        //a
        ma0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        //b
        mb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        //the 0 row
        temp_0 = _mm256_madd_epi16(ma0, mb0);
        temp_1 = _mm256_madd_epi16(ma1, mb0);
        sum0 = _mm256_add_epi32(sum0, temp_0);
        sum1 = _mm256_add_epi32(sum1, temp_1);


        //the 1 row
        mb1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        temp_0 = _mm256_madd_epi16(ma0, mb1);
        temp_1 = _mm256_madd_epi16(ma1, mb1);

        sum2 = _mm256_add_epi32(sum2, temp_0);
        sum3 = _mm256_add_epi32(sum3, temp_1);


        mb2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        temp_0 = _mm256_madd_epi16(ma0, mb2);
        temp_1 = _mm256_madd_epi16(ma1, mb2);

        sum4 = _mm256_add_epi32(sum4, temp_0);
        sum5 = _mm256_add_epi32(sum5, temp_1);

        //the 3 row

        mb3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        temp_0 = _mm256_madd_epi16(ma0, mb3);
        temp_1 = _mm256_madd_epi16(ma1, mb3);
        sum6 = _mm256_add_epi32(sum6, temp_0);
        sum7 = _mm256_add_epi32(sum7, temp_1);

        pa0 += 16;
        pa1 += 16;

        pb0 += 16;
        pb1 += 16;
        pb2 += 16;
        pb3 += 16;
    }

    //store
    __m256i zero = _mm256_setzero_si256();
    __m256 temp_0;
    __m256 temp_1;
    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum2);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    //    pc0[0] = _mm256_extract_epi32(sum0, 0);
    //    pc0[1] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum4 = _mm256_hadd_epi32(sum4, sum6);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    //    pc0[2] = _mm256_extract_epi32(sum4, 0);
    //    pc0[3] = _mm256_extract_epi32(sum4, 1);
    //    printf_intrin_var(sum0);
    //    printf_intrin_var(sum4);
    sum4 = _mm256_blend_epi32(sum0, _mm256_permute4x64_epi64(sum4, 0xc0), 0x0c);
    //    printf_intrin_var(sum4);
    temp_0 = _mm256_broadcast_ps((const __m128*)scale);
    temp_1 = _mm256_cvtepi32_ps(sum4);
    temp_0 = _mm256_mul_ps(temp_0, temp_1);
    __m128 write_128 = _mm256_extractf128_ps(temp_0, 0x00);
    _mm_storeu_ps(pc0, write_128);



    //the 2 row
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, zero, 0x31));


    //the 3 row
    sum5 = _mm256_hadd_epi32(sum5, sum7);
    sum5 = _mm256_hadd_epi32(sum5, zero);
    sum5 = _mm256_add_epi32(sum5, _mm256_permute2x128_si256(sum5, zero, 0x31));

    sum5 = _mm256_blend_epi32(sum1, _mm256_permute4x64_epi64(sum5, 0xc0), 0x0c);
    temp_0 = _mm256_broadcast_ps((const __m128*)scale);
    temp_1 = _mm256_cvtepi32_ps(sum5);
    temp_0 = _mm256_mul_ps(temp_0, temp_1);
    write_128 = _mm256_extractf128_ps(temp_0, 0x00);
    _mm_storeu_ps(pc1, write_128);
}


inline void block2x64_4_kernel_avx2_me_k16_s8s8s8(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int8_t* c, const int32_t ldc, const float* scale_in,
    float* scale_out) {
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;


    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;

    int8_t* pc0 = c;
    int8_t* pc1 = c + 1 * ldc;

    size_t nk = k >> 4; // k / 32
    size_t k_leftover = k - (nk << 4); // k % 32

    __m256i ma0;
    __m256i ma1;

    __m256i mb0;
    __m256i mb1;
    __m256i mb2;
    __m256i mb3;;

    __m256i temp_0;
    __m256i temp_1;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        //b
        mb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        //the 0 row
        temp_0 = _mm256_madd_epi16(ma0, mb0);
        temp_1 = _mm256_madd_epi16(ma1, mb0);
        sum0 = _mm256_add_epi32(sum0, temp_0);
        sum1 = _mm256_add_epi32(sum1, temp_1);


        //the 1 row
        mb1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        temp_0 = _mm256_madd_epi16(ma0, mb1);
        temp_1 = _mm256_madd_epi16(ma1, mb1);

        sum2 = _mm256_add_epi32(sum2, temp_0);
        sum3 = _mm256_add_epi32(sum3, temp_1);


        mb2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        temp_0 = _mm256_madd_epi16(ma0, mb2);
        temp_1 = _mm256_madd_epi16(ma1, mb2);

        sum4 = _mm256_add_epi32(sum4, temp_0);
        sum5 = _mm256_add_epi32(sum5, temp_1);

        //the 3 row

        mb3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        temp_0 = _mm256_madd_epi16(ma0, mb3);
        temp_1 = _mm256_madd_epi16(ma1, mb3);
        sum6 = _mm256_add_epi32(sum6, temp_0);
        sum7 = _mm256_add_epi32(sum7, temp_1);

        pa0 += 16;
        pa1 += 16;

        pb0 += 16;
        pb1 += 16;
        pb2 += 16;
        pb3 += 16;
    }

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum2);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum4 = _mm256_hadd_epi32(sum4, sum6);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc0[2] = _mm256_extract_epi32(sum4, 0);
    pc0[3] = _mm256_extract_epi32(sum4, 1);

    //the 2 row
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_hadd_epi32(sum1, sum3);
    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum1, 0);
    pc1[1] = _mm256_extract_epi32(sum1, 1);

    //the 3 row
    sum5 = _mm256_hadd_epi32(sum5, sum7);
    sum5 = _mm256_hadd_epi32(sum5, zero);
    sum5 = _mm256_add_epi32(sum5, _mm256_permute2x128_si256(sum5, zero, 0x31));

    pc1[2] = _mm256_extract_epi32(sum5, 0);
    pc1[3] = _mm256_extract_epi32(sum5, 1);
}
#if defined(__AVX512F__)
inline __m512i avx512_reduce_4(__m512i& x0, __m512i& x1, __m512i& x2, __m512i& x3) {
    __m512i temp0 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15
    }, x0);
    __m512i temp1 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
    }, x1);
    __m512i temp2 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15
    }, x2);
    __m512i temp3 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
    }, x3);
    temp0 = _mm512_add_epi32(temp0, x0);
    temp1 = _mm512_add_epi32(temp1, x1);
    temp2 = _mm512_add_epi32(temp2, x2);
    temp3 = _mm512_add_epi32(temp3, x3);
    temp0 = _mm512_mask_blend_epi32(0xFF00, temp0, temp1);
    temp2 = _mm512_mask_blend_epi32(0xFF00, temp2, temp3);
    temp1 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 13, 14, 15
    }, temp0);
    temp3 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 13, 14, 15
    }, temp2);
    temp0 = _mm512_add_epi32(temp0, temp1);
    temp2 = _mm512_add_epi32(temp2, temp3);
    temp2 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        0, 1, 2, 3, 0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11
    }, temp2);
    temp0 = _mm512_mask_blend_epi32(0xF0F0, temp0, temp2);
    temp1 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15
    }, temp0);
    temp0 = _mm512_add_epi32(temp0, temp1);
    temp1 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13
    }, temp0);
    temp0 = _mm512_add_epi32(temp0, temp1);
    temp0 = _mm512_permutexvar_epi32((__m512i)(__v16si) {
        0, 8, 4, 12, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    }, temp0);
    return temp0;
}
inline __m512i avx512_loadfp32_int8(const float* ptr, __m512& in_scale) {
    __m512i temp_low = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvt_roundps_epi32(
                           _mm512_mul_ps(_mm512_loadu_ps(ptr), in_scale), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
    __m512i temp_hi = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvt_roundps_epi32(
                          _mm512_mul_ps(_mm512_loadu_ps(ptr + 16), in_scale),
                          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
    temp_hi = _mm512_permutexvar_epi16((__m512i)(__v32hi) {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        14, 15
    }, temp_hi);
    return _mm512_mask_blend_epi16(0xFFFF0000, temp_low, temp_hi);
}

void block4x4_kernel_avx512_me(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32
    __m512i sum0 = _mm512_setzero_si512();
    __m512i sum1 = _mm512_setzero_si512();
    __m512i sum2 = _mm512_setzero_si512();
    __m512i sum3 = _mm512_setzero_si512();
    __m512i sum4 = _mm512_setzero_si512();
    __m512i sum5 = _mm512_setzero_si512();
    __m512i sum6 = _mm512_setzero_si512();
    __m512i sum7 = _mm512_setzero_si512();
    __m512i sum8 = _mm512_setzero_si512();
    __m512i sum9 = _mm512_setzero_si512();
    __m512i sum10 = _mm512_setzero_si512();
    __m512i sum11 = _mm512_setzero_si512();
    __m512i sum12 = _mm512_setzero_si512();
    __m512i sum13 = _mm512_setzero_si512();
    __m512i sum14 = _mm512_setzero_si512();
    __m512i sum15 = _mm512_setzero_si512();

    for (size_t k = 0; k < nk; ++k) {
        __m512i temp0;
        __m512i temp1;
        __m512i temp2;
        __m512i temp3;
        __m512i a0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pa0));
        __m512i a1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pa1));
        __m512i a2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pa2));
        __m512i a3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pa3));

        __m512i b0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb0));
        temp0 = _mm512_madd_epi16(a0, b0);
        temp1 = _mm512_madd_epi16(a1, b0);
        temp2 = _mm512_madd_epi16(a2, b0);
        temp3 = _mm512_madd_epi16(a3, b0);
        sum0 = _mm512_add_epi32(sum0, temp0);
        sum4 = _mm512_add_epi32(sum4, temp1);
        sum8 = _mm512_add_epi32(sum8, temp2);
        sum12 = _mm512_add_epi32(sum12, temp3);

        __m512i b1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb1));
        temp0 = _mm512_madd_epi16(a0, b1);
        temp1 = _mm512_madd_epi16(a1, b1);
        temp2 = _mm512_madd_epi16(a2, b1);
        temp3 = _mm512_madd_epi16(a3, b1);
        sum1 = _mm512_add_epi32(sum1, temp0);
        sum5 = _mm512_add_epi32(sum5, temp1);
        sum9 = _mm512_add_epi32(sum9, temp2);
        sum13 = _mm512_add_epi32(sum13, temp3);

        __m512i b2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb2));
        temp0 = _mm512_madd_epi16(a0, b2);
        temp1 = _mm512_madd_epi16(a1, b2);
        temp2 = _mm512_madd_epi16(a2, b2);
        temp3 = _mm512_madd_epi16(a3, b2);
        sum2 = _mm512_add_epi32(sum2, temp0);
        sum6 = _mm512_add_epi32(sum6, temp1);
        sum10 = _mm512_add_epi32(sum10, temp2);
        sum14 = _mm512_add_epi32(sum14, temp3);

        __m512i b3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb3));
        temp0 = _mm512_madd_epi16(a0, b3);
        temp1 = _mm512_madd_epi16(a1, b3);
        temp2 = _mm512_madd_epi16(a2, b3);
        temp3 = _mm512_madd_epi16(a3, b3);
        sum3 = _mm512_add_epi32(sum3, temp0);
        sum7 = _mm512_add_epi32(sum7, temp1);
        sum11 = _mm512_add_epi32(sum11, temp2);
        sum15 = _mm512_add_epi32(sum15, temp3);


        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;

        pb0 += 32;
        pb1 += 32;
        pb2 += 32;
        pb3 += 32;

    }

    __m512i temp0 = avx512_reduce_4(sum0, sum1, sum2, sum3);
    _mm512_mask_storeu_epi32(pc0, 0x000F, temp0);
    __m512i temp1 = avx512_reduce_4(sum4, sum5, sum6, sum7);
    _mm512_mask_storeu_epi32(pc1, 0x000F, temp1);
    __m512i temp2 = avx512_reduce_4(sum8, sum9, sum10, sum11);
    _mm512_mask_storeu_epi32(pc2, 0x000F, temp2);
    __m512i temp3 = avx512_reduce_4(sum12, sum13, sum14, sum15);
    _mm512_mask_storeu_epi32(pc3, 0x000F, temp3);

    //    printf_intrin_var(temp0);


    //    exit(0);


    //    pc0[0]=_mm512_reduce_add_epi32(sum0);
    //    pc0[1]=_mm512_reduce_add_epi32(sum1);
    //    pc0[2]=_mm512_reduce_add_epi32(sum2);
    //    pc0[3]=_mm512_reduce_add_epi32(sum3);
    //    pc1[0]=_mm512_reduce_add_epi32(sum4);
    //    pc1[1]=_mm512_reduce_add_epi32(sum5);
    //    pc1[2]=_mm512_reduce_add_epi32(sum6);
    //    pc1[3]=_mm512_reduce_add_epi32(sum7);
    //    pc2[0]=_mm512_reduce_add_epi32(sum8);
    //    pc2[1]=_mm512_reduce_add_epi32(sum9);
    //    pc2[2]=_mm512_reduce_add_epi32(sum10);
    //    pc2[3]=_mm512_reduce_add_epi32(sum11);
    //    pc3[0]=_mm512_reduce_add_epi32(sum12);
    //    pc3[1]=_mm512_reduce_add_epi32(sum13);
    //    pc3[2]=_mm512_reduce_add_epi32(sum14);
    //    pc3[3]=_mm512_reduce_add_epi32(sum15);
}

void block4x4_kernel_avx512_me(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, float* c, const int32_t ldc, const float* scale) {
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;

    float* pc0 = c;
    float* pc1 = c + 1 * ldc;
    float* pc2 = c + 2 * ldc;
    float* pc3 = c + 3 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32
    __m512i sum0 = _mm512_setzero_si512();
    __m512i sum1 = _mm512_setzero_si512();
    __m512i sum2 = _mm512_setzero_si512();
    __m512i sum3 = _mm512_setzero_si512();
    __m512i sum4 = _mm512_setzero_si512();
    __m512i sum5 = _mm512_setzero_si512();
    __m512i sum6 = _mm512_setzero_si512();
    __m512i sum7 = _mm512_setzero_si512();
    __m512i sum8 = _mm512_setzero_si512();
    __m512i sum9 = _mm512_setzero_si512();
    __m512i sum10 = _mm512_setzero_si512();
    __m512i sum11 = _mm512_setzero_si512();
    __m512i sum12 = _mm512_setzero_si512();
    __m512i sum13 = _mm512_setzero_si512();
    __m512i sum14 = _mm512_setzero_si512();
    __m512i sum15 = _mm512_setzero_si512();

    for (size_t k = 0; k < nk; ++k) {
        __m512i temp0;
        __m512i temp1;
        __m512i temp2;
        __m512i temp3;
        __m512i a0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pa0));
        __m512i a1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pa1));
        __m512i a2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pa2));
        __m512i a3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pa3));

        __m512i b0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb0));
        temp0 = _mm512_madd_epi16(a0, b0);
        temp1 = _mm512_madd_epi16(a1, b0);
        temp2 = _mm512_madd_epi16(a2, b0);
        temp3 = _mm512_madd_epi16(a3, b0);
        sum0 = _mm512_add_epi32(sum0, temp0);
        sum4 = _mm512_add_epi32(sum4, temp1);
        sum8 = _mm512_add_epi32(sum8, temp2);
        sum12 = _mm512_add_epi32(sum12, temp3);

        __m512i b1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb1));
        temp0 = _mm512_madd_epi16(a0, b1);
        temp1 = _mm512_madd_epi16(a1, b1);
        temp2 = _mm512_madd_epi16(a2, b1);
        temp3 = _mm512_madd_epi16(a3, b1);
        sum1 = _mm512_add_epi32(sum1, temp0);
        sum5 = _mm512_add_epi32(sum5, temp1);
        sum9 = _mm512_add_epi32(sum9, temp2);
        sum13 = _mm512_add_epi32(sum13, temp3);

        __m512i b2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb2));
        temp0 = _mm512_madd_epi16(a0, b2);
        temp1 = _mm512_madd_epi16(a1, b2);
        temp2 = _mm512_madd_epi16(a2, b2);
        temp3 = _mm512_madd_epi16(a3, b2);
        sum2 = _mm512_add_epi32(sum2, temp0);
        sum6 = _mm512_add_epi32(sum6, temp1);
        sum10 = _mm512_add_epi32(sum10, temp2);
        sum14 = _mm512_add_epi32(sum14, temp3);

        __m512i b3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb3));
        temp0 = _mm512_madd_epi16(a0, b3);
        temp1 = _mm512_madd_epi16(a1, b3);
        temp2 = _mm512_madd_epi16(a2, b3);
        temp3 = _mm512_madd_epi16(a3, b3);
        sum3 = _mm512_add_epi32(sum3, temp0);
        sum7 = _mm512_add_epi32(sum7, temp1);
        sum11 = _mm512_add_epi32(sum11, temp2);
        sum15 = _mm512_add_epi32(sum15, temp3);


        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;

        pb0 += 32;
        pb1 += 32;
        pb2 += 32;
        pb3 += 32;

    }

    const __m512 scale_float4 = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0x000F, scale);

    __m512i temp0 = avx512_reduce_4(sum0, sum1, sum2, sum3);
    __m512 wirte_0 = _mm512_cvt_roundepi32_ps(temp0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    wirte_0 = _mm512_mul_ps(wirte_0, scale_float4);
    _mm512_mask_storeu_ps(pc0, 0x000F, wirte_0);

    __m512i temp1 = avx512_reduce_4(sum4, sum5, sum6, sum7);
    __m512 wirte_1 = _mm512_cvt_roundepi32_ps(temp1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    wirte_1 = _mm512_mul_ps(wirte_1, scale_float4);
    _mm512_mask_storeu_ps(pc1, 0x000F, wirte_1);

    __m512i temp2 = avx512_reduce_4(sum8, sum9, sum10, sum11);
    __m512 wirte_2 = _mm512_cvt_roundepi32_ps(temp2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    wirte_2 = _mm512_mul_ps(wirte_2, scale_float4);
    _mm512_mask_storeu_ps(pc2, 0x000F, wirte_2);

    __m512i temp3 = avx512_reduce_4(sum12, sum13, sum14, sum15);
    __m512 wirte_3 = _mm512_cvt_roundepi32_ps(temp3, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    wirte_3 = _mm512_mul_ps(wirte_3, scale_float4);
    _mm512_mask_storeu_ps(pc3, 0x000F, wirte_3);

    //    __m512i temp2=avx512_reduce_4(sum8,sum9,sum10,sum11);
    //    _mm512_mask_storeu_epi32(pc2,0x000F,temp2);
    //    __m512i temp3=avx512_reduce_4(sum12,sum13,sum14,sum15);
    //    _mm512_mask_storeu_epi32(pc3,0x000F,temp3);

    //    printf_intrin_var(temp0);


    //    exit(0);


    //    pc0[0]=_mm512_reduce_add_epi32(sum0);
    //    pc0[1]=_mm512_reduce_add_epi32(sum1);
    //    pc0[2]=_mm512_reduce_add_epi32(sum2);
    //    pc0[3]=_mm512_reduce_add_epi32(sum3);
    //    pc1[0]=_mm512_reduce_add_epi32(sum4);
    //    pc1[1]=_mm512_reduce_add_epi32(sum5);
    //    pc1[2]=_mm512_reduce_add_epi32(sum6);
    //    pc1[3]=_mm512_reduce_add_epi32(sum7);
    //    pc2[0]=_mm512_reduce_add_epi32(sum8);
    //    pc2[1]=_mm512_reduce_add_epi32(sum9);
    //    pc2[2]=_mm512_reduce_add_epi32(sum10);
    //    pc2[3]=_mm512_reduce_add_epi32(sum11);
    //    pc3[0]=_mm512_reduce_add_epi32(sum12);
    //    pc3[1]=_mm512_reduce_add_epi32(sum13);
    //    pc3[2]=_mm512_reduce_add_epi32(sum14);
    //    pc3[3]=_mm512_reduce_add_epi32(sum15);
}

void block4x4_kernel_avx512_me(
    const int32_t k, const float* a,  const int32_t lda, const float scale_a,
    const int8_t* b, const int32_t ldb, float* c, const int32_t ldc, const float* scale) {
    //    LOG(INFO)<<"in_scale = "<<scale_a;
    const float* pa0 = a;
    const float* pa1 = pa0 + 1 * lda;
    const float* pa2 = pa0 + 2 * lda;
    const float* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;

    float* pc0 = c;
    float* pc1 = c + 1 * ldc;
    float* pc2 = c + 2 * ldc;
    float* pc3 = c + 3 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32
    __m512i sum0 = _mm512_setzero_si512();
    __m512i sum1 = _mm512_setzero_si512();
    __m512i sum2 = _mm512_setzero_si512();
    __m512i sum3 = _mm512_setzero_si512();
    __m512i sum4 = _mm512_setzero_si512();
    __m512i sum5 = _mm512_setzero_si512();
    __m512i sum6 = _mm512_setzero_si512();
    __m512i sum7 = _mm512_setzero_si512();
    __m512i sum8 = _mm512_setzero_si512();
    __m512i sum9 = _mm512_setzero_si512();
    __m512i sum10 = _mm512_setzero_si512();
    __m512i sum11 = _mm512_setzero_si512();
    __m512i sum12 = _mm512_setzero_si512();
    __m512i sum13 = _mm512_setzero_si512();
    __m512i sum14 = _mm512_setzero_si512();
    __m512i sum15 = _mm512_setzero_si512();
    __m512 in_scale = _mm512_set1_ps(scale_a);

    for (size_t k = 0; k < nk; ++k) {
        __m512i temp0;
        __m512i temp1;
        __m512i temp2;
        __m512i temp3;

        __m512i a0 = avx512_loadfp32_int8(pa0, in_scale);
        __m512i a1 = avx512_loadfp32_int8(pa1, in_scale);
        __m512i a2 = avx512_loadfp32_int8(pa2, in_scale);
        __m512i a3 = avx512_loadfp32_int8(pa3, in_scale);

        __m512i b0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb0));
        temp0 = _mm512_madd_epi16(a0, b0);
        temp1 = _mm512_madd_epi16(a1, b0);
        temp2 = _mm512_madd_epi16(a2, b0);
        temp3 = _mm512_madd_epi16(a3, b0);
        sum0 = _mm512_add_epi32(sum0, temp0);
        sum4 = _mm512_add_epi32(sum4, temp1);
        sum8 = _mm512_add_epi32(sum8, temp2);
        sum12 = _mm512_add_epi32(sum12, temp3);

        __m512i b1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb1));
        temp0 = _mm512_madd_epi16(a0, b1);
        temp1 = _mm512_madd_epi16(a1, b1);
        temp2 = _mm512_madd_epi16(a2, b1);
        temp3 = _mm512_madd_epi16(a3, b1);
        sum1 = _mm512_add_epi32(sum1, temp0);
        sum5 = _mm512_add_epi32(sum5, temp1);
        sum9 = _mm512_add_epi32(sum9, temp2);
        sum13 = _mm512_add_epi32(sum13, temp3);

        __m512i b2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb2));
        temp0 = _mm512_madd_epi16(a0, b2);
        temp1 = _mm512_madd_epi16(a1, b2);
        temp2 = _mm512_madd_epi16(a2, b2);
        temp3 = _mm512_madd_epi16(a3, b2);
        sum2 = _mm512_add_epi32(sum2, temp0);
        sum6 = _mm512_add_epi32(sum6, temp1);
        sum10 = _mm512_add_epi32(sum10, temp2);
        sum14 = _mm512_add_epi32(sum14, temp3);

        __m512i b3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)pb3));
        temp0 = _mm512_madd_epi16(a0, b3);
        temp1 = _mm512_madd_epi16(a1, b3);
        temp2 = _mm512_madd_epi16(a2, b3);
        temp3 = _mm512_madd_epi16(a3, b3);
        sum3 = _mm512_add_epi32(sum3, temp0);
        sum7 = _mm512_add_epi32(sum7, temp1);
        sum11 = _mm512_add_epi32(sum11, temp2);
        sum15 = _mm512_add_epi32(sum15, temp3);


        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;

        pb0 += 32;
        pb1 += 32;
        pb2 += 32;
        pb3 += 32;

    }

    const __m512 scale_float4 = _mm512_mask_loadu_ps(_mm512_setzero_ps(), 0x000F, scale);

    __m512i temp0 = avx512_reduce_4(sum0, sum1, sum2, sum3);
    __m512 wirte_0 = _mm512_cvt_roundepi32_ps(temp0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    wirte_0 = _mm512_mul_ps(wirte_0, scale_float4);
    _mm512_mask_storeu_ps(pc0, 0x000F, wirte_0);

    __m512i temp1 = avx512_reduce_4(sum4, sum5, sum6, sum7);
    __m512 wirte_1 = _mm512_cvt_roundepi32_ps(temp1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    wirte_1 = _mm512_mul_ps(wirte_1, scale_float4);
    _mm512_mask_storeu_ps(pc1, 0x000F, wirte_1);

    __m512i temp2 = avx512_reduce_4(sum8, sum9, sum10, sum11);
    __m512 wirte_2 = _mm512_cvt_roundepi32_ps(temp2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    wirte_2 = _mm512_mul_ps(wirte_2, scale_float4);
    _mm512_mask_storeu_ps(pc2, 0x000F, wirte_2);

    __m512i temp3 = avx512_reduce_4(sum12, sum13, sum14, sum15);
    __m512 wirte_3 = _mm512_cvt_roundepi32_ps(temp3, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    wirte_3 = _mm512_mul_ps(wirte_3, scale_float4);
    _mm512_mask_storeu_ps(pc3, 0x000F, wirte_3);

    //    __m512i temp2=avx512_reduce_4(sum8,sum9,sum10,sum11);
    //    _mm512_mask_storeu_epi32(pc2,0x000F,temp2);
    //    __m512i temp3=avx512_reduce_4(sum12,sum13,sum14,sum15);
    //    _mm512_mask_storeu_epi32(pc3,0x000F,temp3);

    //    printf_intrin_var(temp0);


    //    exit(0);


    //    pc0[0]=_mm512_reduce_add_epi32(sum0);
    //    pc0[1]=_mm512_reduce_add_epi32(sum1);
    //    pc0[2]=_mm512_reduce_add_epi32(sum2);
    //    pc0[3]=_mm512_reduce_add_epi32(sum3);
    //    pc1[0]=_mm512_reduce_add_epi32(sum4);
    //    pc1[1]=_mm512_reduce_add_epi32(sum5);
    //    pc1[2]=_mm512_reduce_add_epi32(sum6);
    //    pc1[3]=_mm512_reduce_add_epi32(sum7);
    //    pc2[0]=_mm512_reduce_add_epi32(sum8);
    //    pc2[1]=_mm512_reduce_add_epi32(sum9);
    //    pc2[2]=_mm512_reduce_add_epi32(sum10);
    //    pc2[3]=_mm512_reduce_add_epi32(sum11);
    //    pc3[0]=_mm512_reduce_add_epi32(sum12);
    //    pc3[1]=_mm512_reduce_add_epi32(sum13);
    //    pc3[2]=_mm512_reduce_add_epi32(sum14);
    //    pc3[3]=_mm512_reduce_add_epi32(sum15);
}
/**
* b must packed
*/
inline void avx512_s8s8s32_gemm_4x4_packed(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    const int m_block = 4;
    const int n_block = 4;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0 ," << m;
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0 ," << n;
#if USE_OMP_IN_INTRINSIC_PACKED_FC
    #pragma omp parallel for schedule(static)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];

            int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block4x4_kernel_avx512_me(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
        }
    }
}

inline void avx512_s8s8s32_gemm_4x4_packed(
    const int32_t m, const int32_t n, const int32_t k,
    const float* a, const int32_t lda, const float scale_a,
    const int8_t* b, const int32_t ldb,
    float* c, const int32_t ldc, const float* scale) {
    const int m_block = 4;
    const int n_block = 4;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0 ," << m;
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0 ," << n;
#if USE_OMP_IN_INTRINSIC_PACKED_FC
    #pragma omp parallel for schedule(static)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const float* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];

            float* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block4x4_kernel_avx512_me(k, a_ptr, lda, scale_a, b_ptr, ldb, c_ptr, ldc, scale);
        }
    }
}

inline void avx512_s8s8s32_gemm_4x4_packed(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    float* c, const int32_t ldc, const float* scale) {
    const int m_block = 4;
    const int n_block = 4;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0 ," << m;
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0 ," << n;
#if USE_OMP_IN_INTRINSIC_PACKED_FC
    #pragma omp parallel for schedule(static)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];

            float* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block4x4_kernel_avx512_me(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, scale);
        }
    }
}

#endif

/**
* b must packed
*/
inline void avx_s8s8s32_gemm_2x4_packed(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    const int m_block = 2;
    const int n_block = 4;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0";
#if USE_OMP_IN_INTRINSIC_PACKED_FC
    #pragma omp parallel for schedule(static)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];
            int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            //            block4x2_kernel_avx2_me(k,a_ptr,lda,b_ptr,ldb,c_ptr,ldc,1);
            //            block4x2_kernel_avx2_me_k16(k,a_ptr,lda,b_ptr,ldb,c_ptr,ldc,1);
            block2x4_kernel_avx2_me_k16(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
        }
    }
}

inline void avx_s8s8s32_gemm_2x4_packed_omp_packed(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    const int m_block = 2;
    const int n_block = 4;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0";
#if USE_OMP_IN_INTRINSIC_PACKED_FC
    #pragma omp parallel for schedule(static)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];
            int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block2x4_kernel_avx2_me_k16_packed(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
        }
    }
}

/**
* b must packed
*/
inline void avx_s8s8s32_gemm_2x4_packed_omp(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    const int m_block = 2;
    const int n_block = 4;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0";
    //    auto ker = [&](const int ithr, const int nthr) {
    //        for (int mbi = 0; mbi < mb; mbi++) {
    //            for (int nbi = 0; nbi < nb; nbi++) {
    //                const int8_t* a_ptr = &a[mbi * m_block * lda];
    //                const int8_t* b_ptr = &b[nbi * n_block * ldb];
    //                int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
    //                block2x4_kernel_avx2_me_k16(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
    //            }
    //        }
    //    };
    ////#pragma omp parallel
    //    {
    //        ker(anakin_get_thread_num(), anakin_get_num_threads());
    //    }

#if USE_OMP_IN_INTRINSIC_PACKED_FC
#pragma omp parallel for schedule(static) if (anakin_get_max_threads() > 1)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];
            int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block2x4_kernel_avx2_me_k16(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
        }
    }


}

inline void avx_s8s8s32_gemm_1x8_packed_omp(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    const int m_block = 1;
    const int n_block = 8;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0";
#if USE_OMP_IN_INTRINSIC_PACKED_FC
#pragma omp parallel for schedule(static)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];
            int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block1x8_kernel_avx2_me_k16(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
        }
    }
}
#if 0
template <DataType A_Dtype, DataType B_Dtype, DataType C_Dtype>
SaberStatus PackedFC<A_Dtype, B_Dtype, C_Dtype>::init(int n, int k, int8_t* weights) {
    CHECK_EQ(k % 16, 0);
    _inner_weights.re_alloc(Shape({1, 1, n, k}), AK_INT8);
    int8_t* out_ptr = static_cast<int8_t*>(_inner_weights.mutable_data());

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            int in_index = i * n + j;
            int out_index = j * k + i;
            out_ptr[out_index] = weights[in_index];
        }
    }

    jit::jit_int8_packed_fc_config_t  int8_generate_config;
    int8_generate_config.m_block_size = 2;
    int8_generate_config.n_block_size = 4;
    int8_generate_config.k_block_number = k / 16;
    _packed_gemm = new jit::jit_s8s8s32_packed_gemm(int8_generate_config);
    _packed_gemm->dump_code(_packed_gemm->getCode());
    return SaberSuccess;
}
#endif
template <DataType A_Dtype, DataType B_Dtype, DataType C_Dtype>
SaberStatus PackedFC<A_Dtype, B_Dtype, C_Dtype>::init(int n, int k, Tensor<X86>& weights_tensor,
        float input_scale,
        float output_scale, PackedFCAlg alg) {
    _alg = alg;

    if (B_Dtype == AK_INT8) {
        LOG(INFO) << "init = " << alg;

        if (alg == DotAdd) {
            CHECK_EQ(k % 16, 0);
            packed_weights_k2(_inner_weights, weights_tensor, n, k, 8);
            return SaberSuccess;
        } else if (alg == DotReductionPacked) {
            CHECK_EQ(k % 16, 0);
            packed_weights_transpose_k(_inner_weights, weights_tensor, n, k, 4, 16);
            return SaberSuccess;
        } else if (alg == DotSplitK) {
            CHECK_EQ(k % 2, 0);
            packed_weights_k2(_inner_weights, weights_tensor, n, k, 64);
            return SaberSuccess;
        } else {
            CHECK_EQ(k % 16, 0);
            _inner_weights.re_alloc(Shape({1, 1, n, k}), AK_INT8);
            int8_t* out_ptr = static_cast<int8_t*>(_inner_weights.mutable_data());

            const int8_t* weights = static_cast<const int8_t*>(weights_tensor.data());

            for (int i = 0; i < k; i++) {
                for (int j = 0; j < n; j++) {
                    int in_index = i * n + j;
                    int out_index = j * k + i;
                    out_ptr[out_index] = weights[in_index];
                }
            }

            jit::jit_int8_packed_fc_config_t int8_generate_config;
            int8_generate_config.m_block_size = 2;
            int8_generate_config.n_block_size = 4;
            int8_generate_config.k_block_number = k / 16;
            //        _packed_gemm = new jit::jit_s8s8s32_packed_gemm(int8_generate_config);
            //        _packed_gemm->dump_code(_packed_gemm->getCode());
            return SaberSuccess;
        }
    } else {
        CHECK_EQ(weights_tensor.get_dtype(), AK_FLOAT);
        _inner_weights.re_alloc(Shape({1, 1, n, k}), AK_INT8);
        Tensor<X86> temp_tensor(Shape({1, 1, n, k}), AK_INT8);
        int8_t* out_ptr = static_cast<int8_t*>(_inner_weights.mutable_data());
        utils::ScaleUtils::scale_gemm_xw_weights_to_nchw_host(temp_tensor, weights_tensor);
        const int8_t* weights = static_cast<const int8_t*>(temp_tensor.data());

        //    printf_pointer(weights,n*k);
        //    printf_pointer(temp_tensor.get_scale().data(),n);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                int in_index = i * n + j;
                int out_index = j * k + i;
                out_ptr[out_index] = weights[in_index];
            }
        }

        _inner_weights.set_scale(temp_tensor.get_scale());
        auto weights_scales = _inner_weights.get_scale();
        _scale.clear();

        for (auto weights_scale : weights_scales) {
            _scale.push_back(input_scale * weights_scale / output_scale);
        }

        return SaberSuccess;
    }

}

#if 0
SaberStatus PackedFC::dispatch(const int m, const int n, const int k, const int8_t* a,
                               int* c) {
    const int8_t* b = static_cast<const int8_t*>(_inner_weights.data());

    //    if (m == 1 || m % 2 == 1) {
    //        avx_s8s8s32_gemm_1x8_packed_omp(m, n, k, a, k, b, k, c, n);
    //    } else {
    //        avx_s8s8s32_gemm_2x4_packed_omp(m, n, k, a, k, b, k, c, n);
    //    }
    //    LOG(INFO)<<"m = "<<m<<","<<n<<","<<k<<", c ptr = "<<c;
    avx512_s8s8s32_gemm_4x4_packed(m, n, k, a, k, b, k, c, n);
    return SaberSuccess;

    //    const int m_block = 2;
    //    const int n_block = 4;
    //    const int lda=k;
    //    const int ldb=k;
    //    const int ldc=n;
    //    int mb = m / m_block;
    //    int nb = n / n_block;
    //    int m_remainder = m % m_block;
    //    int n_remainder = n % n_block;
    //            CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    //            CHECK_EQ(n_remainder, 0) << "only support remainder = 0";
    //
    //    for (int mbi = 0; mbi < mb; mbi++) {
    //        for (int nbi = 0; nbi < nb; nbi++) {
    //            const int8_t* a_ptr = &a[mbi * m_block * lda];
    //            const int8_t* b_ptr = &b[nbi * n_block * ldb];
    //            int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
    //            jit::jit_int8_packed_fc_call_t  int8_config;
    //            int8_config.lda = lda;
    //            int8_config.ldb = ldb;
    //            int8_config.ldc = ldc;
    //            int8_config.weights = b_ptr;
    //            int8_config.src = a_ptr;
    //            int8_config.output_data = c_ptr;
    //            int8_config.k_block = k / 16;
    //            _packed_gemm->jit_ker(&int8_config);
    //        }
    //    }

    return SaberSuccess;
}

SaberStatus PackedFC::dispatch(const int m, const int n, const int k, const int8_t* a,
                               float* c) {

    const int8_t* b = static_cast<const int8_t*>(_inner_weights.data());
    const float* sclae = _inner_weights.get_scale().data();
    const int m_block = 2;
    const int n_block = 4;
    const int lda = k;
    const int ldb = k;
    const int ldc = n;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0";
#if USE_OMP_IN_INTRINSIC_PACKED_FC
#pragma omp parallel for schedule(static)
#endif

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];
            float* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block2x4_kernel_avx2_me_k16_pad_s8s8fp32(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, sclae);
        }
    }

    return SaberSuccess;
}

SaberStatus PackedFC::dispatch(const int m, const int n, const int k, const Tensor<X86>& a,
                               float* c) {
    if (jit::mayiuse(jit::avx512_core) && a.get_dtype() == AK_INT8 && _scale.size() > 0) {
        const int8_t* a_scale_ptr = static_cast<const int8_t*>(_scale_inputs.data());
        const int8_t* b = static_cast<const int8_t*>(_inner_weights.data());
        const float* sclae = _scale.data();
        //    printf_pointer(sclae,_scale.size());
        const int m_block = 4;
        const int n_block = 4;
        const int lda = k;
        const int ldb = k;
        const int ldc = n;
        int mb = m / m_block;
        int nb = n / n_block;
        int m_remainder = m % m_block;
        int n_remainder = n % n_block;
        CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
        CHECK_EQ(n_remainder, 0) << "only support remainder = 0";
        LOG(INFO) << "it is scale gemm ";
#if USE_OMP_IN_INTRINSIC_PACKED_FC
        #pragma omp parallel for schedule(static)
#endif

        for (int mbi = 0; mbi < mb; mbi++) {
            for (int nbi = 0; nbi < nb; nbi++) {
                const int8_t* a_ptr = &a_scale_ptr[mbi * m_block * lda];
                const int8_t* b_ptr = &b[nbi * n_block * ldb];
                float* c_ptr = &c[mbi * m_block * n + nbi * n_block];
                block4x4_kernel_avx512_scale_me(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, sclae);
            }
        }

    } else {
        CHECK_EQ(a.get_dtype(), AK_FLOAT);
        utils::try_expand_tensor(_scale_inputs, a.valid_shape());
        utils::ScaleUtils::scale_fp32_int8(_scale_inputs, a);
        const int8_t* a_scale_ptr = static_cast<const int8_t*>(_scale_inputs.data());
        const int8_t* b = static_cast<const int8_t*>(_inner_weights.data());
        const float* sclae = _scale.data();
        //    printf_pointer(sclae,_scale.size());
        const int m_block = 2;
        const int n_block = 4;
        const int lda = k;
        const int ldb = k;
        const int ldc = n;
        int mb = m / m_block;
        int nb = n / n_block;
        int m_remainder = m % m_block;
        int n_remainder = n % n_block;
        CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
        CHECK_EQ(n_remainder, 0) << "only support remainder = 0";

#if USE_OMP_IN_INTRINSIC_PACKED_FC
#pragma omp parallel for schedule(static)
#endif

        for (int mbi = 0; mbi < mb; mbi++) {
            for (int nbi = 0; nbi < nb; nbi++) {
                const int8_t* a_ptr = &a_scale_ptr[mbi * m_block * lda];
                const int8_t* b_ptr = &b[nbi * n_block * ldb];
                float* c_ptr = &c[mbi * m_block * n + nbi * n_block];
                //                    LOG(INFO)<<"are you ok";
                //            printf_pointer(a_ptr,2*k);

                block2x4_kernel_avx2_me_k16_pad_s8s8fp32(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, sclae);
            }
        }
    }

    return SaberSuccess;
}

#endif

template < DataType datatype>
struct MyDataTrait {
    typedef __invalid_type Dtype;
};
template <>
struct MyDataTrait<AK_FLOAT> {
    typedef float Dtype;
};
template <>
struct MyDataTrait<AK_INT32> {
    typedef int Dtype;
};
template <>
struct MyDataTrait<AK_INT8> {
    typedef int8_t Dtype;
};
template <>
struct MyDataTrait<AK_UINT8> {
    typedef uint8_t Dtype;
};
template <>
SaberStatus PackedFC<AK_INT8, AK_INT8, AK_INT32>::dispatch(const int m, const int n, const int k,
        const Tensor<X86>& tensor_a,
        Tensor<X86>& tensor_c) {
    CHECK_EQ(tensor_a.get_dtype(), AK_INT8);
    CHECK(tensor_c.get_dtype() == AK_INT32 || tensor_c.get_dtype() == AK_FLOAT);
    const int8_t* b = static_cast<const int8_t*>(_inner_weights.data());
    const int8_t* a = static_cast<const int8_t*>(tensor_a.data());
    int* c = static_cast<int*>(tensor_c.mutable_data());

    if (_alg == DotAdd) {
#if defined(__AVX2__) and defined(__FMA__)
        //        avx_s8s8s32_gemm_mx8_packed_dot_add(m, n, k, a, k, b, k, c, n);
        avx_s8s8s32_gemm_4x8_packed_dot_add(m, n, k, a, k, b, k, c, n);
#else
        LOG(FATAL) << "not impl";
#endif
    } else if (_alg == DotReductionPacked) {
#if defined(__AVX2__) and defined(__FMA__)
        avx_s8s8s32_gemm_2x4_packed_omp_packed(m, n, k, a, k, b, k, c, n);
#else
        LOG(FATAL) << "not impl";
#endif
    } else if (_alg == DotSplitK) {
#if defined(__AVX2__) and defined(__FMA__)
        avx_s8s8s32_gemm_4x64_packed_split_k(m, n, k, a, k, b, k, c, n);
#else
        LOG(FATAL) << "not impl";
#endif
    } else {
#if defined(__AVX512F__)
        avx512_s8s8s32_gemm_4x4_packed(m, n, k, a, k, b, k, c, n);
#elif defined(__AVX2__) and defined(__FMA__)
        avx_s8s8s32_gemm_2x4_packed_omp(m, n, k, a, k, b, k, c, n);
#else
        LOG(FATAL) << "not impl";
#endif
    }

    return SaberSuccess;
}
template <>
SaberStatus PackedFC<AK_FLOAT, AK_FLOAT, AK_FLOAT>::dispatch(const int m, const int n, const int k,
        const Tensor<X86>& tensor_a,
        Tensor<X86>& tensor_c) {
    CHECK_EQ(tensor_a.get_dtype(), AK_FLOAT);
    CHECK_EQ(tensor_c.get_dtype(), AK_FLOAT);
    CHECK_EQ(_scale.size(), n);
    CHECK_EQ(tensor_a.get_scale().size(), 1);
    const float scale_a = 1.f / tensor_a.get_scale()[0];
    const float* sclae = _scale.data();
    const int8_t* b = static_cast<const int8_t*>(_inner_weights.data());
    const float* a = static_cast<const float*>(tensor_a.data());
    float* c = static_cast<float*>(tensor_c.mutable_data());
#if defined(__AVX512F__)
    avx512_s8s8s32_gemm_4x4_packed(m, n, k, a, k, scale_a, b, k, c, n, sclae);
#else
    LOG(FATAL) << "not impl";
#endif
    return SaberSuccess;
}
//template <>
//SaberStatus PackedFC<AK_INT8,AK_INT8,AK_FLOAT>::dispatch(const int m, const int n, const int k, const Tensor<X86>& tensor_a,
//                                                           Tensor<X86> &tensor_c) {
//            CHECK_EQ(_scale.size(),n);
//            CHECK_EQ(tensor_a.get_scale().size(),1);
//    const float scale_a=1.f/tensor_a.get_scale()[0];
//    const float* sclae=_scale.data();
//    const int8_t* b = static_cast<const int8_t*>(_inner_weights.data());
//    const int8_t * a= static_cast<const int8_t *>(tensor_a.data());
//    float* c= static_cast<float *>(tensor_c.mutable_data());
//    avx512_s8s8s32_gemm_4x4_packed(m, n, k, a, k,scale_a, b, k, c, n,sclae);
//    return SaberSuccess;
//}

template class PackedFC<AK_FLOAT, AK_FLOAT, AK_FLOAT>;
template class PackedFC<AK_INT8, AK_INT8, AK_INT32>;
//template class PackedFC<AK_INT8,AK_INT8,AK_FLOAT>;
#else

template <>
SaberStatus PackedFC<AK_INT8, AK_INT8, AK_INT32>::
init(int n, int k, Tensor<X86>& weights_tensor,float input_scale,float output_scale,PackedFCAlg alg) {
    LOG(FATAL) << "not impl";
    return SaberSuccess;
}

template <>
SaberStatus PackedFC<AK_FLOAT, AK_FLOAT, AK_FLOAT>::
init(int n, int k, Tensor<X86>& weights_tensor,float input_scale,float output_scale,PackedFCAlg alg) {
    LOG(FATAL) << "not impl";
    return SaberSuccess;
}

template <>
SaberStatus PackedFC<AK_INT8, AK_INT8, AK_INT32>::
dispatch(const int m, const int n, const int k, const Tensor<X86>& tensor_a,
         Tensor<X86>& tensor_c) {
    LOG(FATAL) << "not impl";
    return SaberSuccess;
};

template <>
SaberStatus PackedFC<AK_FLOAT, AK_FLOAT, AK_FLOAT>::
dispatch(const int m, const int n, const int k, const Tensor<X86>& tensor_a,
         Tensor<X86>& tensor_c) {
    LOG(FATAL) << "not impl";
    return SaberSuccess;
};

#endif

}
}
