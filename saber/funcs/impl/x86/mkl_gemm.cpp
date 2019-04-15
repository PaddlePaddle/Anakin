#include "saber/funcs/impl/x86/mkl_gemm.h"
#include "saber/funcs/timer.h"
#include "debug.h"
namespace anakin {

namespace saber {
#define MKL_GEMM_TIMER 0
template <>
SaberStatus MklDnnGemm<float, float, float >::init(
    const bool trans_a, const bool trans_b,
    const int m, const int n, const int k,
    Context<X86> ctx, const float* ptr_b, MKLGemmMode gemm_mode) {
    _gemm_mode = gemm_mode;
    _lda = (!trans_a) ? k : m;
    _ldb = (!trans_b) ? n : k;
    _ldc = n;
    _m = m;
    _n = n;
    _k = k;
    _trans_a = trans_a ? 'T' : 'N';
    _trans_b = trans_b ? 'T' : 'N';

    if (gemm_mode == PACKED_MKLGEMM) {
        if (_weights_packed_ptr_fp32 != nullptr) {
            cblas_sgemm_free(_weights_packed_ptr_fp32);
        }

        _weights_packed_ptr_fp32 = cblas_sgemm_alloc(CblasBMatrix, m, n, k);

        cblas_sgemm_pack(CblasRowMajor,
                         CblasBMatrix,
                         trans_b ? CblasTrans : CblasNoTrans,
                         m, n, k,
                         1.0,
                         ptr_b, n,
                         _weights_packed_ptr_fp32);
    }

    return SaberSuccess;
}

template <>
SaberStatus MklDnnGemm<float, float, float>::dispatch(
    const float alpha, const float beta, int m,
    const float* ptr_a, const float* ptr_b, float* ptr_c) {
    CHECK(ptr_a != nullptr);
    CHECK(ptr_b != nullptr);
    CHECK(ptr_c != nullptr);

    //    LOG(INFO)<<"it is mkldnn gemm";
#if MKL_GEMM_TIMER
    Context<X86> ctx(0, 0, 0);
    SaberTimer<X86> timer;
    timer.start(ctx);
#endif

    if (_gemm_mode == PACKED_MKLGEMM) {
        //        LOG(INFO)<<"MklDnnGemm dispatch "<<_m<<","<<_n<<","<<_k;
        cblas_sgemm_compute(CblasRowMajor,
                            CblasNoTrans,
                            CblasPacked,
                            m, _n, _k,
                            ptr_a, _k,
                            _weights_packed_ptr_fp32, _n,
                            beta,
                            ptr_c, _n);
    } else {
        CBLAS_TRANSPOSE trans_a =
            (_trans_a == 'T') ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE trans_b =
            (_trans_b == 'T') ? CblasTrans : CblasNoTrans;
        CHECK(ptr_b != nullptr);
        cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, _n, _k, alpha, ptr_a, _lda, ptr_b, _ldb, beta,
                    ptr_c, _ldc);
    }

#if MKL_GEMM_TIMER
    timer.end(ctx);
    double ms = timer.get_average_ms();
    double work_load = (double)_m * _n * _k * 2;
    double speed = work_load / ms / 1000.0 / 1000.0;
    LOG(INFO) << "mkldnn_gemm_f32f32f32 [" << _gemm_mode << "] " << _m << "," << _n << "," << _k << ","
              << ms << "," << speed;
#endif

    return SaberSuccess;
}

template <>
SaberStatus MklDnnGemm<char, char, int>::init(
    const bool trans_a, const bool trans_b,
    const int m, const int n, const int k,
    Context<X86> ctx, const char* ptr_b, MKLGemmMode gemm_mode) {
    _gemm_mode = gemm_mode;
    _lda = (!trans_a) ? k : m;
    _ldb = (!trans_b) ? n : k;
    _ldc = n;
    _m = m;
    _n = n;
    _k = k;
    _trans_a = trans_a ? 'T' : 'N';
    _trans_b = trans_b ? 'T' : 'N';

    auto s8_a = true;
    auto packed_b = gemm_mode == PACKED_MKLGEMM;
    char oc_mode = 'R';
    auto ocsize = oc_mode == 'R' ? n : oc_mode == 'C' ? m : 1;
    _oc_offset.re_alloc(Shape({1, 1, 1, ocsize}), AK_INT32);
    fill_tensor_const(_oc_offset, 0);
    auto status = _packed_s8s8s32_gemm.init(ptr_b, _oc_offset.data(), &_s8s8s32_handle, oc_mode,
                                            m, n, k, 0, 0, s8_a, packed_b, trans_a, trans_b,
                                            0.f, 1.f, _lda, _ldb, _ldc);
    CHECK_EQ(status, SaberSuccess);

    return SaberSuccess;
}

template <>
SaberStatus MklDnnGemm<char, char, int>::dispatch(
    const float alpha, const float beta, int m,
    const char* ptr_a, const char* ptr_b, int* ptr_c) {
    CHECK(ptr_a != nullptr);
    CHECK(ptr_b != nullptr);
    CHECK(ptr_c != nullptr);

    if (_gemm_mode == PACKED_MKLGEMM) {
        auto status = _packed_s8s8s32_gemm.execute(_s8s8s32_handle, m,  ptr_a, ptr_c);
        CHECK_EQ(status, SaberSuccess);
    } else {
        auto status = _packed_s8s8s32_gemm.execute(_s8s8s32_handle, m,  ptr_a, ptr_c, ptr_b);
        CHECK_EQ(status, SaberSuccess);
    }

    return SaberSuccess;
}

template <>
SaberStatus MklDnnGemm<int8_t, int8_t, int>::init(
    const bool trans_a, const bool trans_b,
    const int m, const int n, const int k,
    Context<X86> ctx, const int8_t* ptr_b, MKLGemmMode gemm_mode) {
    _gemm_mode = gemm_mode;
    _lda = (!trans_a) ? k : m;
    _ldb = (!trans_b) ? n : k;
    _ldc = n;
    _m = m;
    _n = n;
    _k = k;
    _trans_a = trans_a ? 'T' : 'N';
    _trans_b = trans_b ? 'T' : 'N';

    auto s8_a = true;
    auto packed_b = gemm_mode == PACKED_MKLGEMM;
    char oc_mode = 'R';
    auto ocsize = oc_mode == 'R' ? n : oc_mode == 'C' ? m : 1;
    _oc_offset.re_alloc(Shape({1, 1, 1, ocsize}), AK_INT32);
    fill_tensor_const(_oc_offset, 0);
    auto status = _packed_s8s8s32_gemm.init(ptr_b, _oc_offset.data(), &_s8s8s32_handle, oc_mode,
                                            m, n, k, 0, 0, s8_a, packed_b, trans_a, trans_b,
                                            0.f, 1.f, _lda, _ldb, _ldc);
    CHECK_EQ(status, SaberSuccess);


    return SaberSuccess;
}

template <>
SaberStatus MklDnnGemm<int8_t, int8_t, int>::dispatch(
    const float alpha, const float beta, int m,
    const int8_t* ptr_a, const int8_t* ptr_b, int* ptr_c) {
    CHECK(ptr_a != nullptr);
    CHECK(ptr_c != nullptr);
#if MKL_GEMM_TIMER
    Context<X86> ctx(0, 0, 0);
    SaberTimer<X86> timer;
    timer.start(ctx);
#endif

    if (_gemm_mode == PACKED_MKLGEMM) {
        auto status = _packed_s8s8s32_gemm.execute(_s8s8s32_handle, m, ptr_a, ptr_c);
        CHECK_EQ(status, SaberSuccess);
    } else {
        auto status = _packed_s8s8s32_gemm.execute(_s8s8s32_handle, m, ptr_a, ptr_c, ptr_b);
        CHECK_EQ(status, SaberSuccess);
    }

#if MKL_GEMM_TIMER
    timer.end(ctx);
    double ms = timer.get_average_ms();
    double work_load = (double)_m * _n * _k * 2;
    double speed = work_load / ms / 1000.0 / 1000.0;
    LOG(INFO) << "mkldnn_gemm_s8s8s32 " << _m << "," << _n << "," << _k << "," << ms << "," << speed;
#endif
    return SaberSuccess;
}

template <>
SaberStatus MklDnnGemm<uint8_t, int8_t, int>::init(
    const bool trans_a, const bool trans_b,
    const int m, const int n, const int k,
    Context<X86> ctx, const int8_t* ptr_b, MKLGemmMode gemm_mode) {
    _gemm_mode = gemm_mode;
    _lda = (!trans_a) ? k : m;
    _ldb = (!trans_b) ? n : k;
    _ldc = n;
    _m = m;
    _n = n;
    _k = k;
    _trans_a = trans_a ? 'T' : 'N';
    _trans_b = trans_b ? 'T' : 'N';

    auto s8_a = false;
    auto packed_b = gemm_mode == PACKED_MKLGEMM;
    char oc_mode = 'R';
    auto ocsize = oc_mode == 'R' ? n : oc_mode == 'C' ? m : 1;
    _oc_offset.re_alloc(Shape({1, 1, 1, ocsize}), AK_INT32);
    fill_tensor_const(_oc_offset, 0);
    auto status = _packed_s8s8s32_gemm.init(ptr_b, _oc_offset.data(), &_s8s8s32_handle, oc_mode,
                                            m, n, k, 0, 0, s8_a, packed_b, trans_a, trans_b,
                                            0.f, 1.f, _lda, _ldb, _ldc);
    CHECK_EQ(status, SaberSuccess);


    return SaberSuccess;
}

template <>
SaberStatus MklDnnGemm<uint8_t, int8_t, int>::dispatch(
    const float alpha, const float beta, int m,
    const uint8_t* ptr_a, const int8_t* ptr_b, int* ptr_c) {
    CHECK(ptr_a != nullptr);
    CHECK(ptr_c != nullptr);
#if MKL_GEMM_TIMER
    Context<X86> ctx(0, 0, 0);
    SaberTimer<X86> timer;
    timer.start(ctx);
#endif

    if (_gemm_mode == PACKED_MKLGEMM) {
        auto status = _packed_s8s8s32_gemm.execute(_s8s8s32_handle, m,  ptr_a, ptr_c);
        CHECK_EQ(status, SaberSuccess);
    } else {
        auto status = _packed_s8s8s32_gemm.execute(_s8s8s32_handle, m, ptr_a, ptr_c, ptr_b);
        CHECK_EQ(status, SaberSuccess);
    }

#if MKL_GEMM_TIMER
    timer.end(ctx);
    double ms = timer.get_average_ms();
    double work_load = (double)_m * _n * _k * 2;
    double speed = work_load / ms / 1000.0 / 1000.0;
    LOG(INFO) << "mkldnn_gemm_s8s8s32 " << _m << "," << _n << "," << _k << "," << ms << "," << speed;
#endif
    return SaberSuccess;
}

template <>
SaberStatus MklDnnGemm<unsigned char, char, int>::init(
    const bool trans_a, const bool trans_b,
    const int m, const int n, const int k,
    Context<X86> ctx, const char* ptr_b, MKLGemmMode gemm_mode) {
    _gemm_mode = gemm_mode;
    _lda = (!trans_a) ? k : m;
    _ldb = (!trans_b) ? n : k;
    _ldc = n;
    _m = m;
    _n = n;
    _k = k;
    _trans_a = trans_a ? 'T' : 'N';
    _trans_b = trans_b ? 'T' : 'N';

    auto s8_a = false;
    auto packed_b = gemm_mode == PACKED_MKLGEMM;
    char oc_mode = 'R';
    auto ocsize = oc_mode == 'R' ? n : oc_mode == 'C' ? m : 1;
    _oc_offset.re_alloc(Shape({1, 1, 1, ocsize}), AK_INT32);
    fill_tensor_const(_oc_offset, 0);
    auto status = _packed_s8s8s32_gemm.init(ptr_b, _oc_offset.data(), &_s8s8s32_handle, oc_mode,
                                            m, n, k, 0, 0, s8_a, packed_b, trans_a, trans_b,
                                            0.f, 1.f, _lda, _ldb, _ldc);
    CHECK_EQ(status, SaberSuccess);


    return SaberSuccess;
}

template <>
SaberStatus MklDnnGemm<unsigned char, char, int>::dispatch(
    const float alpha, const float beta, int m,
    const unsigned char* ptr_a, const char* ptr_b, int* ptr_c) {
    CHECK(ptr_a != nullptr);
    CHECK(ptr_c != nullptr);
#if MKL_GEMM_TIMER
    Context<X86> ctx(0, 0, 0);
    SaberTimer<X86> timer;
    timer.start(ctx);
#endif

    if (_gemm_mode == PACKED_MKLGEMM) {
        auto status = _packed_s8s8s32_gemm.execute(_s8s8s32_handle, m,  ptr_a, ptr_c);
        CHECK_EQ(status, SaberSuccess);
    } else {
        auto status = _packed_s8s8s32_gemm.execute(_s8s8s32_handle, m,  ptr_a, ptr_c, ptr_b);
        CHECK_EQ(status, SaberSuccess);
    }

#if MKL_GEMM_TIMER
    timer.end(ctx);
    double ms = timer.get_average_ms();
    double work_load = (double)_m * _n * _k * 2;
    double speed = work_load / ms / 1000.0 / 1000.0;
    LOG(INFO) << "mkldnn_gemm_s8s8s32 " << _m << "," << _n << "," << _k << "," << ms << "," << speed;
#endif
    return SaberSuccess;
}





}
}