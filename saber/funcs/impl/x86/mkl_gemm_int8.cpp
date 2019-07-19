#include "saber/funcs/impl/x86/mkl_gemm_int8.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {

template<>
SaberStatus MKLGEMM<AK_INT8>::mem_a_s82u8(const int8_t* src, size_t length) {
    if (src == nullptr) {
        LOG(FATAL) << "wrong empty pointer !";
        return SaberInvalidValue;
    }

    utils::try_expand_tensor(_inner_u8_matrix_a, length);

    uint8_t* inner_u8_ptr = static_cast<uint8_t*>(_inner_u8_matrix_a.data());
    uint8_t* scr_pointer = (uint8_t*)src;
#pragma omp parallel for

    for (auto i = 0; i < length; i++) {
        inner_u8_ptr[i] = scr_pointer[i] + 128;
    }

    return SaberSuccess;
}

template<>
void* MKLGEMM<AK_INT8>::mem_oc_s8a_compute(void* handle) {
    if (handle == nullptr) {
        LOG(FATAL) << "wrong empty pointer !";
        return nullptr;
    }

    auto args = static_cast<gemm_param*>(handle);
    auto b_mem = static_cast<const int8_t*>(args->matrix_b);

    if (b_mem == nullptr) {
        LOG(FATAL) << "wrong empty pointer !";
        return nullptr;
    }

    if (args->s8_a) {
        auto dim_k = args->k;
        auto dim_n = args->n;
        auto ob = args->ob;
        auto dst = static_cast<int32_t*>(calloc(dim_n, sizeof(int32_t)));
        auto oc_mem = args->matrix_oc
                      ? static_cast<const int32_t*>(args->matrix_oc)
                      : nullptr;
        auto fix_oc = oc_mem ? oc_mem[0] : 0;
        auto alpha = args->alpha;
        auto scale = args->alpha * -128;

        auto thread_num = omp_max_thread;

        if (dim_n <= 2) {
            thread_num = 1;
        } else if (dim_n < omp_max_thread) {
            thread_num = dim_n;
        }

        if (args->oc_mode == 'F') {
            if (args->trans_b) {
#pragma omp parallel for collapse(1) num_threads(thread_num)
                for (auto i = 0; i < dim_n; i++) {
                    int32_t b_dim_k_sum = 0;
#pragma omp simd
                    for (auto j = 0; j < dim_k; j++) {
                        b_dim_k_sum += b_mem[i * dim_k + j] + ob;
                    }

                    dst[i] += scale * b_dim_k_sum + fix_oc;
                }
            } else {
                for (auto i = 0; i < dim_k; i++) {
                    if (i == 0) {
#pragma omp parallel for collapse(1) num_threads(thread_num)
                        for (auto j = 0; j < dim_n; j++) {
                            dst[j] += scale * (b_mem[i * dim_n + j] + ob) + fix_oc;
                        }
                    } else {
#pragma omp parallel for collapse(1) num_threads(thread_num)
                        for (auto j = 0; j < dim_n; j++) {
                            dst[j] += scale * (b_mem[i * dim_n + j] + ob);
                        }
                    }

                }
            }
        } else if (args->oc_mode == 'R') {
            if (args->trans_b) {
#pragma omp parallel for collapse(1) num_threads(thread_num)
                for (auto i = 0; i < dim_n; i++) {
                    int32_t b_dim_k_sum = 0;
                    #pragma omp simd

                    for (auto j = 0; j < dim_k; j++) {
                        b_dim_k_sum += b_mem[i * dim_k + j] + ob;
                    }

                    dst[i] += scale * b_dim_k_sum + oc_mem[i];
                }
            } else {
                for (auto i = 0; i < dim_k; i++) {
                    if (i == 0) {
#pragma omp parallel for collapse(1) num_threads(thread_num)
                        for (auto j = 0; j < dim_n; j++) {
                            dst[j] += scale * (b_mem[i * dim_n + j] + ob) + oc_mem[j];
                        }
                    } else {
#pragma omp parallel for collapse(1) num_threads(thread_num)
                        for (auto j = 0; j < dim_n; j++) {
                            dst[j] += scale * (b_mem[i * dim_n + j] + ob);
                        }
                    }
                }
            }
        } else if (args->oc_mode == 'C') {
            if (args->trans_b) {
#pragma omp parallel for collapse(1) num_threads(thread_num)
                for (auto i = 0; i < dim_n; i++) {
                    int32_t b_dim_k_sum = 0;
#pragma omp simd
                    for (auto j = 0; j < dim_k; j++) {
                        b_dim_k_sum += b_mem[i * dim_k + j] + ob;
                    }

                    dst[i] += scale * b_dim_k_sum;
                }
            } else {
                for (auto i = 0; i < dim_k; i++) {
#pragma omp parallel for collapse(1) num_threads(thread_num)
                    for (auto j = 0; j < dim_n; j++) {
                        dst[j] += scale * (b_mem[i * dim_n + j] + ob);
                    }
                }
            }
        }

        return dst;
    }

    return nullptr;
}
template<>
void MKLGEMM<AK_INT8>::add_mem_oc_s8a(char oc_mode, const int* oc_mem,
                                      const void* b_in, int8_t ob,
                                      size_t m, size_t k, size_t n, float alpha, bool trans_b) {
    CHECK_EQ(oc_mode, 'R') << "only support C offset";
    CHECK_EQ(trans_b, false) << "only support no trans b now";
    auto thread_num = omp_max_thread;

    if (m <= 2) {
        thread_num = 1;
    } else if (m < omp_max_thread) {
        thread_num = m;
    }

    int8_t* b_mem = (int8_t*)b_in;
    int scale = (int)round(alpha * -128);
    int* oc_offset = (int*)_inner_c_offset.mutable_data();
    memset(oc_offset, 0, sizeof(int)*n);

    for (auto i = 0; i < k; i++) {
        if (i == 0) {
#pragma omp parallel for collapse(1) num_threads(thread_num)
            for (auto j = 0; j < n; j++) {
                oc_offset[j] += scale * (b_mem[i * n + j] + ob) + oc_mem[j];
            }
        } else {
#pragma omp parallel for collapse(1) num_threads(thread_num)
            for (auto j = 0; j < n; j++) {
                oc_offset[j] += scale * (b_mem[i * n + j] + ob);
            }
        }
    }

}
template<>
SaberStatus MKLGEMM<AK_INT8>::add_mem_oc_s8a(bool a_s82u8, char oc_mode,
        const void* in, void* out,
        size_t dim_m, size_t dim_n) {
    if (a_s82u8 && oc_mode == 'C') {
        if (in == nullptr || out == nullptr) {
            LOG(FATAL) << "wrong empty pointer !";
            return SaberInvalidValue;
        }

        auto src = static_cast<const int32_t*>(in);
        auto dst = static_cast<int32_t*>(out);

        auto thread_num = omp_max_thread;

        if (dim_m <= 2) {
            thread_num = 1;
        } else if (dim_m < omp_max_thread) {
            thread_num = dim_m;
        }

#pragma omp parallel for collapse(1) num_threads(thread_num)
        for (auto h = 0; h < dim_m; h++) {
#pragma omp simd
            for (auto w = 0; w < dim_n; w++) {
                dst[h * dim_n + w] += src[w];
            }
        }
    } else{
        DLOG(INFO)<<"do nothing";
    }

    return SaberSuccess;
}

template<>
void* MKLGEMM<AK_INT8>::pack_mem(const void* mem_in,
                                 const bool pack_b,
                                 const bool trans,
                                 const size_t m,
                                 const size_t n,
                                 const size_t k,
                                 const size_t stride,
                                 const float alpha) {
    CHECK_EQ(mem_in != nullptr, true) << "wrong empty pointer !";

    void* mem_out = nullptr;
    auto identifier = pack_b ? CblasBMatrix : CblasAMatrix;
    auto need_trans = trans ? CblasTrans : CblasNoTrans;
    auto length = cblas_gemm_s8u8s32_pack_get_size(identifier, m, n, k);
    mem_out = malloc(length);
    cblas_gemm_s8u8s32_pack(CblasRowMajor,
                            identifier,
                            need_trans,
                            m,
                            n,
                            k,
                            mem_in,
                            stride,
                            mem_out);

    return mem_out;
}

template<>
SaberStatus MKLGEMM<AK_INT8>::execute(const void* mem_a,
                                      const void* mem_b,
                                      const void* mem_oc,
                                      void* mem_c,
                                      const bool s8_a,
                                      const size_t m,
                                      const size_t n,
                                      const size_t k,
                                      const int8_t oa,
                                      const int8_t ob,
                                      const size_t lda,
                                      const size_t ldb,
                                      const size_t ldc,
                                      const bool pack_b,
                                      const bool trans_a,
                                      const bool trans_b,
                                      const float beta,
                                      const float alpha,
                                      const char offset_mode) {
    auto status = execute_check(mem_a, mem_b, mem_oc, mem_c, oa, ob, offset_mode);

    if (status != SaberSuccess) {
        LOG(ERROR) << "check failed";
        return status;
    }

    auto dst = static_cast<int32_t*>(mem_c);
    auto offset = static_cast<const int32_t*>(mem_oc);
    auto a_trans = trans_a ? CblasTrans : CblasNoTrans;
    auto b_trans = trans_b ? CblasTrans : CblasNoTrans;
    auto b_mode = pack_b ? (CBLAS_TRANSPOSE)CblasPacked : b_trans;
    auto oc_mode = CblasFixOffset;

    if (offset_mode == 'F') {
        oc_mode = CblasFixOffset;
    } else if (offset_mode == 'R') {
        oc_mode = CblasRowOffset;
    } else if (offset_mode == 'C') {
        oc_mode = CblasColOffset;
    }


    if (pack_b) {

        cblas_gemm_s8u8s32_compute(CblasRowMajor,
                                   a_trans,
                                   b_mode,
                                   oc_mode,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   mem_a,
                                   lda,
                                   oa,
                                   mem_b,
                                   ldb,
                                   ob,
                                   beta,
                                   dst,
                                   ldc,
                                   offset);

    } else {
        cblas_gemm_s8u8s32(CblasRowMajor,
                           a_trans,
                           b_trans,
                           oc_mode,
                           m,
                           n,
                           k,
                           alpha,
                           mem_a,
                           lda,
                           oa,
                           mem_b,
                           ldb,
                           ob,
                           beta,
                           dst,
                           ldc,
                           offset);
    }

    return SaberSuccess;
};

template<>
SaberStatus MKLGEMM<AK_INT8>::execute(const void* handle, const int m, const void* a_matrix, void* c_matrix,
                                      const void* b_matrix) {
    auto args = static_cast<const gemm_param*>(handle);

    auto status = SaberSuccess;

    CHECK(args->pack_b || b_matrix != nullptr);
    ((gemm_param*)(handle))->m=m;

    if (args->s8_a) {
        mem_a_s82u8(static_cast<const int8_t* >(a_matrix), args->m * args->k);

        if (args->pack_b) {
            CHECK_EQ(args->oc_mode, 'R');
            status = execute(_inner_u8_matrix_a.data(), args->pack_b ? args->packed_mem : b_matrix,
                             args->oc_mem_s8a,
                             c_matrix, args->s8_a, args->m, args->n, args->k, args->oa,
                             args->ob, args->lda, args->ldb, args->ldc, args->pack_b,
                             args->trans_a, args->trans_b, args->beta, args->alpha,
                             args->s8a_oc_mode);
        } else {
            CHECK_EQ(args->oc_mode, 'R');
            add_mem_oc_s8a(args->oc_mode, (int*)args->matrix_oc, b_matrix, args->ob, args->m, args->k, args->n,
                           args->alpha, args->trans_b);
            status = execute(_inner_u8_matrix_a.data(), args->pack_b ? args->packed_mem : b_matrix,
                             (int*)_inner_c_offset.mutable_data(),
                             c_matrix, args->s8_a, args->m, args->n, args->k, args->oa,
                             args->ob, args->lda, args->ldb, args->ldc, args->pack_b,
                             args->trans_a, args->trans_b, args->beta, args->alpha,
                             args->s8a_oc_mode);
        }
    } else {
        status = execute(a_matrix, args->pack_b ? args->packed_mem : b_matrix,
                         args->matrix_oc, c_matrix, args->s8_a, args->m, args->n, args->k,
                         args->oa, args->ob, args->lda, args->ldb, args->ldc, args->pack_b,
                         args->trans_a, args->trans_b, args->beta, args->alpha, args->oc_mode);
    }

    if (status != SaberSuccess) {
        return status;
    }

    return SaberSuccess;
}

} // namespace saber
} // namespace anakin