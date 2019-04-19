#include "saber/core/context.h"
#include "saber/funcs/gemm.h"
#include "saber/funcs/timer.h"
#include "saber/core/tensor.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "conv_func_helper.h"

#if defined(USE_X86_PLACE)
#include "saber/funcs/impl/x86/mkl_gemm.h"
#include "saber/funcs/impl/x86/mkl_packed_int8_gemm.h"
#include <emmintrin.h>
#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#endif

#include <vector>
#include "debug.h"

using namespace anakin::saber;
#define CLEAR_CACHE 0


#ifdef USE_X86_PLACE

void flush_tensor_cache_out(Tensor<X86>& tensor) {
#ifdef USE_X86_PLACE
    char* ptr = static_cast<char*>(tensor.data());
    size_t amount = tensor.valid_size() * tensor.get_dtype_size();

    for (size_t i = 0; i < amount; i += 32) {
        _mm_clflush(ptr + i);
    }

#endif
}
#endif

void gemm_check(const int m, const int n, const int k,
                const float* a, const float* b, float* c,
                const float alpha, const float beta,
                const bool trans_a, const bool trans_b) {
    if (!trans_a && !trans_b) {
        int lda = k;
        int ldb = n;
        int ldc = n;

        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;

                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[m_i * lda + k_i] * b[k_i * ldb + n_i];
                }
            }
        }
    } else if (!trans_a && trans_b) {
        int lda = k;
        int ldb = k;
        int ldc = n;

        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;

                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[m_i * lda + k_i] * b[n_i * ldb + k_i];
                }
            }
        }
    } else if (trans_a && !trans_b) {
        int lda = m;
        int ldb = n;
        int ldc = n;

        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;

                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[k_i * lda + m_i] * b[k_i * ldb + n_i];
                }
            }
        }
    } else {
        int lda = m;
        int ldb = k;
        int ldc = n;

        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;

                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[k_i * lda + m_i] * b[n_i * ldb + k_i];
                }
            }
        }
    }

}

#if defined(USE_X86_PLACE)
template <typename TargetType, typename TargetType_H>
void test_gemm_result_mkldnn(int m, int n, int k, bool trans_a, bool trans_b,
                             MKLGemmMode gemm_mode = NORMAL_MKLGEMM) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;

    Context<TargetType> ctx1(0, 1, 0);
    MklDnnGemm<float, float, float> gemm_vender;


    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({m, k}, Layout_HW);
    Shape b_shape({k, n}, Layout_HW);
    Shape c_shape({m, n}, Layout_HW);

    a_dev.re_alloc(a_shape, AK_FLOAT);
    b_dev.re_alloc(b_shape, AK_FLOAT);
    c_dev.re_alloc(c_shape, AK_FLOAT);

    a_host.re_alloc(a_shape, AK_FLOAT);
    b_host.re_alloc(b_shape, AK_FLOAT);
    c_host.re_alloc(c_shape, AK_FLOAT);
    c_check.re_alloc(c_shape, AK_FLOAT);
    fill_tensor_rand(a_dev, -10.f, 10.f);
    fill_tensor_rand(b_dev, -10.f, 10.f);
    a_host.copy_from(a_dev);
    b_host.copy_from(b_dev);
    SaberTimer<TargetType> vender_time, saber_time;
    int ts = 1000;
    int warm_up = 100;

    SaberStatus vender_status = gemm_vender.init(trans_a, trans_b, 1, n, k, ctx1,
                                static_cast<float*>(b_dev.data()), gemm_mode);

    if (vender_status == SaberSuccess) {
        gemm_vender.dispatch(alpha, beta, m,
                             (const float*) a_dev.data(),
                             (const float*) b_dev.data(),
                             (float*) c_dev.mutable_data());

        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        c_host.copy_from(c_dev);
        gemm_check(m, n, k, (const float*) a_host.data(), (const float*) b_host.data(),
                   (float*) c_check.mutable_data(),
                   alpha, beta, trans_a, trans_b);
        double max_ratio = 0.f, max_diff = 0.f;
        tensor_cmp_host((const float*) c_check.data(), (const float*) c_host.data(),
                        c_check.valid_size(), max_ratio, max_diff);

        if (max_ratio > 1e-3) {
            print_tensor(a_dev);
            print_tensor(b_dev);
            print_tensor_valid(c_check);
            print_tensor_valid(c_host);
            LOG(FATAL) << "VENDER: FAIL!!!! max_ratio = " << max_ratio << " max_diff: " << max_diff
                       << "m = " << m << " n = " << n << " k = " << k;
        }

        for (int t = 0; t < warm_up; t++) {
            gemm_vender.dispatch(alpha, beta, m,
                                 (const float*) a_dev.data(),
                                 (const float*) b_dev.data(),
                                 (float*) c_dev.mutable_data());
        }

        for (int t = 0; t < ts; ++t) {
            vender_time.start(ctx1);
            gemm_vender.dispatch(alpha, beta, m,
                                 (const float*) a_dev.data(),
                                 (const float*) b_dev.data(),
                                 (float*) c_dev.mutable_data());
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            vender_time.end(ctx1);
        }
    }

    double work = (double)m * n * k * 2;
    double vender_time_ms = (vender_status == SaberSuccess ? vender_time.get_average_ms() : 1e10);
    double vender_speed = work / vender_time_ms / 1000.0 / 1000.0;
    LOG(INFO) << "mkldnn " << m << "," << n << "," << k << "::" << "gops " << vender_speed << ", ms = "
              << vender_time_ms;
    //    LOG(INFO) << "Vender time: " << (vender_status == SaberSuccess ? vender_time.get_average_ms() : 0)
    //              << "ms ,speed = " << vender_speed << "gfloat/s";
}

template <typename TargetType, typename TargetType_H>
void test_gemm_result_mkl_warp(int m, int n, int k, bool trans_a, bool trans_b) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;

    Context<TargetType> ctx1(0, 1, 0);
    PackedMKLInt8Gemm gemm_vender;
    float input_max = 3.f;

    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({1, 1, m, k}, Layout_NCHW);
    Shape b_shape({1, 1, k, n}, Layout_NCHW);
    Shape c_shape({1, 1, m, n}, Layout_NCHW);

    a_dev.re_alloc(a_shape, AK_FLOAT);
    b_dev.re_alloc(b_shape, AK_FLOAT);
    c_dev.re_alloc(c_shape, AK_FLOAT);

    a_host.re_alloc(a_shape, AK_FLOAT);
    b_host.re_alloc(b_shape, AK_FLOAT);
    c_host.re_alloc(c_shape, AK_FLOAT);
    c_check.re_alloc(c_shape, AK_FLOAT);
//    fill_tensor_rand(a_dev, -input_max, input_max);
//    fill_tensor_rand(b_dev, -input_max, input_max);
    fill_tensor_const(a_dev,input_max);
    fill_tensor_const(b_dev,input_max);
    a_host.copy_from(a_dev);
    b_host.copy_from(b_dev);
    SaberTimer<TargetType> vender_time, saber_time;
    int ts = 1000;
    int warm_up = 100;

    a_dev.set_scale({input_max / 127.f});
    SaberStatus vender_status = gemm_vender.init(trans_a, trans_b, 1, n, k, b_dev,
                                a_dev.get_scale()[0]);

    if (vender_status == SaberSuccess) {
        gemm_vender.dispatch(alpha, beta, m,
                             a_dev,
                             c_dev, nullptr);

        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        c_host.copy_from(c_dev);
        gemm_check(m, n, k, (const float*) a_host.data(), (const float*) b_host.data(),
                   (float*) c_check.mutable_data(),
                   alpha, beta, trans_a, trans_b);
        double max_ratio = 0.f, max_diff = 0.f;

        tensor_cmp_host_mlu((const float*)c_host.data(), (const float*)c_check.data(),
                            c_check.valid_size(), max_ratio, max_diff);

        if (max_ratio > 0.1) {
            write_tensorfile(c_dev, "output_dev");
            write_tensorfile(c_check, "check_host");
            LOG(FATAL) << "VENDER: FAIL!!!! max_ratio = " << max_ratio << " max_diff: " << max_diff
                       << "m = " << m << " n = " << n << " k = " << k;
        } else {
            LOG(INFO) << "passed " << max_ratio;
        }

        for (int t = 0; t < warm_up; t++) {
            gemm_vender.dispatch(alpha, beta, m,
                                 a_dev,
                                 c_dev, nullptr);
        }

        for (int t = 0; t < ts; ++t) {
            vender_time.start(ctx1);
            gemm_vender.dispatch(alpha, beta, m,
                                 a_dev,
                                 c_dev, nullptr);
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            vender_time.end(ctx1);
        }
    }

    double work = (double)m * n * k * 2;
    double vender_time_ms = (vender_status == SaberSuccess ? vender_time.get_average_ms() : 1e10);
    double vender_speed = work / vender_time_ms / 1000.0 / 1000.0;
    LOG(INFO) << "mkldnn " << m << "," << n << "," << k << "::" << "gops " << vender_speed << ", ms = "
              << vender_time_ms;
    //    LOG(INFO) << "Vender time: " << (vender_status == SaberSuccess ? vender_time.get_average_ms() : 0)
    //              << "ms ,speed = " << vender_speed << "gfloat/s";
}

#endif

template <typename TargetType, typename TargetType_H>
void test_gemm_result(int m, int n, int k, bool trans_a, bool trans_b) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;

    Context<TargetType> ctx1(0, 1, 0);
    Gemm<TargetType, VENDER_IMPL, float> gemm_vender;
    Gemm<TargetType, SABER_IMPL, float> gemm_saber;
    SaberStatus vender_status = gemm_vender.init(trans_a, trans_b, m, n, k, ctx1);
    SaberStatus saber_status = gemm_saber.init(trans_a, trans_b, m, n, k, ctx1);

    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({m, k}, Layout_HW);
    Shape b_shape({k, n}, Layout_HW);
    Shape c_shape({m, n}, Layout_HW);

    a_dev.re_alloc(a_shape, AK_FLOAT);
    b_dev.re_alloc(b_shape, AK_FLOAT);
    c_dev.re_alloc(c_shape, AK_FLOAT);

    a_host.re_alloc(a_shape, AK_FLOAT);
    b_host.re_alloc(b_shape, AK_FLOAT);
    c_host.re_alloc(c_shape, AK_FLOAT);
    c_check.re_alloc(c_shape, AK_FLOAT);
    fill_tensor_rand(a_dev, -10.f, 10.f);
    fill_tensor_rand(b_dev, -10.f, 10.f);
    a_host.copy_from(a_dev);
    b_host.copy_from(b_dev);
    SaberTimer<TargetType> vender_time, saber_time;

    int ts = 300;
    int warm_up = 50;

    if (vender_status == SaberSuccess) {
        gemm_vender.dispatch(alpha, beta,
                             (const float*) a_dev.data(),
                             (const float*) b_dev.data(),
                             (float*) c_dev.mutable_data());
        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        c_host.copy_from(c_dev);
        gemm_check(m, n, k, (const float*) a_host.data(), (const float*) b_host.data(),
                   (float*) c_check.mutable_data(),
                   alpha, beta, trans_a, trans_b);
        double max_ratio = 0.f, max_diff = 0.f;
        tensor_cmp_host((const float*) c_check.data(), (const float*) c_host.data(),
                        c_check.valid_size(), max_ratio, max_diff);

        if (max_ratio > 1e-3) {
            print_tensor_valid(c_check);
            print_tensor_valid(c_host);
            LOG(FATAL) << "VENDER: FAIL!!!! max_ratio = " << max_ratio << " max_diff: " << max_diff
                       << "m = " << m << " n = " << n << " k = " << k;
        }

        for (int t = 0; t < warm_up; t++) {
            gemm_vender.dispatch(alpha, beta,
                                 (const float*) a_dev.data(),
                                 (const float*) b_dev.data(),
                                 (float*) c_dev.mutable_data());
        }


        for (int t = 0; t < ts; ++t) {
#if CLEAR_CACHE
            flush_tensor_cache_out(a_dev);
            flush_tensor_cache_out(b_dev);
            flush_tensor_cache_out(c_dev);
#endif
            vender_time.start(ctx1);
            gemm_vender.dispatch(alpha, beta,
                                 (const float*) a_dev.data(),
                                 (const float*) b_dev.data(),
                                 (float*) c_dev.mutable_data());

            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            vender_time.end(ctx1);
        }



    }

    if (saber_status == SaberSuccess) {
        gemm_saber.dispatch(alpha, beta,
                            (const float*) a_dev.data(),
                            (const float*) b_dev.data(),
                            (float*) c_dev.mutable_data());
        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        c_host.copy_from(c_dev);
        gemm_check(m, n, k, (const float*) a_host.data(), (const float*) b_host.data(),
                   (float*) c_check.mutable_data(),
                   alpha, beta, trans_a, trans_b);
        double max_ratio = 0.f, max_diff = 0.f;
        tensor_cmp_host((const float*) c_check.data(), (const float*) c_host.data(),
                        c_check.valid_size(), max_ratio, max_diff);

        if (max_ratio > 1e-3) {
            print_tensor_valid(c_check);
            print_tensor_valid(c_host);
            LOG(FATAL) << "SABER: FAIL!!!! max_ratio = " << max_ratio << " max_diff: " << max_diff
                       << "m = " << m << " n = " << n << " k = " << k;
        }

        for (int t = 0; t < ts; ++t) {
            saber_time.start(ctx1);
            gemm_saber.dispatch(alpha, beta,
                                (const float*) a_dev.data(),
                                (const float*) b_dev.data(),
                                (float*) c_dev.mutable_data());
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            saber_time.end(ctx1);
        }
    }

    double work = (double)m * n * k * 2;
    double vender_time_ms = (vender_status == SaberSuccess ? vender_time.get_average_ms() : 1e10);
    double saber_time_ms = (saber_status == SaberSuccess ? saber_time.get_average_ms() : 1e10);
    double vender_speed = work / vender_time_ms / 1000.0 / 1000.0;
    double saber_speed = work / saber_time_ms / 1000.0 / 1000.0;
    LOG(INFO) << "mkl " << m << "," << n << "," << k << "::" << "gops " << vender_speed << ", ms = " <<
              vender_time_ms << " : ";
    //    LOG(INFO) << "Vender time: " << (vender_status == SaberSuccess ? vender_time.get_average_ms() : 0)
    //              << "ms ,speed = " << vender_speed << "gfloat/s"
    //              << " ,Saber time: " << (saber_status == SaberSuccess ? saber_time.get_average_ms() : 0)
    //              << " ms,speed = " << saber_speed << " gfloat/s";
}

void gemv_check(const int m, const int n, const float* a, const float* b, float* c,
                const float alpha, const float beta,
                const bool trans) {
    if (!trans) {
        int lda = n;

        for (int m_i = 0; m_i < m; ++m_i) {
            c[m_i] *= beta;

            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i] += alpha * a[m_i * lda + n_i] * b[n_i];
            }
        }
    } else {
        int lda = n;

        for (int n_i = 0; n_i < n; ++n_i) {
            c[n_i] *= beta;

            for (int m_i = 0; m_i < m; ++m_i) {
                c[n_i] += alpha * a[m_i * lda + n_i] * b[m_i];
            }
        }
    }
}




template <typename TargetType, typename TargetType_H>
void test_gemv_result(int m, int n, bool trans) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;
    int incx = 1;
    int incy = 1;
    Context<TargetType> ctx1(0, 1, 0);
    Gemv<TargetType, VENDER_IMPL, float> gemv_vender;
    Gemv<TargetType, SABER_IMPL, float> gemv_saber;
    SaberStatus vender_status = gemv_vender.init(trans, m, n, incx, incy, ctx1);
    SaberStatus saber_status = gemv_saber.init(trans, m, n, incx, incy, ctx1);

    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({m, n}, Layout_HW);
    Shape b_shape({(trans ? m : n)}, Layout_W);
    Shape c_shape({(trans ? n : m)}, Layout_W);

    a_dev.re_alloc(a_shape, AK_FLOAT);
    b_dev.re_alloc(b_shape, AK_FLOAT);
    c_dev.re_alloc(c_shape, AK_FLOAT);

    a_host.re_alloc(a_shape, AK_FLOAT);
    b_host.re_alloc(b_shape, AK_FLOAT);
    c_host.re_alloc(c_shape, AK_FLOAT);
    c_check.re_alloc(c_shape, AK_FLOAT);
    fill_tensor_rand(a_dev, -10.f, 10.f);
    fill_tensor_rand(b_dev, -10.f, 10.f);
    a_host.copy_from(a_dev);
    b_host.copy_from(b_dev);

    SaberTimer<TargetType> vender_time, saber_time;
    int ts = 1000;

    if (vender_status == SaberSuccess) {
        gemv_vender.dispatch(alpha, beta,
                             (const float*) a_dev.data(),
                             (const float*) b_dev.data(),
                             (float*) c_dev.mutable_data());
        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        c_host.copy_from(c_dev);
        gemv_check(m, n, (const float*) a_host.data(), (const float*) b_host.data(),
                   (float*) c_check.mutable_data(),
                   alpha, beta, trans);
        double max_ratio = 0.f, max_diff = 0.f;
        tensor_cmp_host((const float*) c_check.data(), (const float*) c_host.data(),
                        c_check.valid_size(), max_ratio, max_diff);

        if (max_ratio > 1e-3) {
            print_tensor_valid(a_host);
            print_tensor_valid(b_host);
            print_tensor_valid(c_check);
            print_tensor_valid(c_host);
            LOG(FATAL) << "VENDER: FAIL!!!! max_ratio = " << max_ratio << " max_diff: " << max_diff
                       << "m = " << m << " n = " << n;
        }

        for (int t = 0; t < ts; ++t) {
            vender_time.start(ctx1);
            gemv_vender.dispatch(alpha, beta,
                                 (const float*) a_dev.data(),
                                 (const float*) b_dev.data(),
                                 (float*) c_dev.mutable_data());
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            vender_time.end(ctx1);
        }
    }

    if (saber_status == SaberSuccess) {
        gemv_saber.dispatch(alpha, beta,
                            (const float*) a_dev.data(),
                            (const float*) b_dev.data(),
                            (float*) c_dev.mutable_data());
        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        c_host.copy_from(c_dev);
        gemv_check(m, n, (const float*) a_host.data(), (const float*) b_host.data(),
                   (float*) c_check.mutable_data(),
                   alpha, beta, trans);
        double max_ratio = 0.f, max_diff = 0.f;
        tensor_cmp_host((const float*) c_check.data(), (const float*) c_host.data(),
                        c_check.valid_size(), max_ratio, max_diff);

        if (max_ratio > 1e-3) {
            print_tensor_valid(c_check);
            print_tensor_valid(c_host);
            LOG(FATAL) << "SABER: FAIL!!!! max_ratio = " << max_ratio << " max_diff: " << max_diff
                       << "m = " << m << " n = " << n;
        }

        for (int t = 0; t < ts; ++t) {
            saber_time.start(ctx1);
            gemv_saber.dispatch(alpha, beta,
                                (const float*) a_dev.data(),
                                (const float*) b_dev.data(),
                                (float*) c_dev.mutable_data());
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            saber_time.end(ctx1);
        }
    }

    LOG(INFO) << "Vender time: " << (vender_status == SaberSuccess ? vender_time.get_average_ms() : 0)
              << " ms Saber time: " << (saber_status == SaberSuccess ? saber_time.get_average_ms() : 0)
              << " ms";
}

TEST(TestSaberFunc, test_vender_gemm_float) {
#if defined(USE_X86_PLACE)

    //    std::vector<int> m_v = {5, 100, 150, 200, 250, 300};
    //    std::vector<int> n_v = {5, 100, 150, 200, 250, 300};
    //    std::vector<int> k_v = {5, 100, 150, 200, 250, 300};
    //    std::vector<int> trans_a_v{false};
    //    std::vector<int> trans_b_v{false};
    //
    //    for (auto m : m_v)
    //    for (auto n : n_v)
    //    for (auto k : k_v)
    //    for (auto trans_a : trans_a_v)
    //    for (auto trans_b : trans_b_v) {
    //
    //#ifdef USE_CUDA
    //        test_gemm_result<NV, NVHX86>(m, n, k, trans_a, trans_b);
    //#endif
    //
    //#ifdef USE_X86_PLACE
    //        test_gemm_result<X86, X86>(m, n, k, trans_a, trans_b);
    //#endif
    //    }

    //    test_gemm_result_mkldnn<X86,X86>(2,3,4,false,false);

    //    int n = 4096, k = 1024;
    //
    //    for (int m : {
    //                1, 2, 3, 4, 12, 16
    //            }) {
    ////        test_gemm_result<X86, X86>(m, n, k, false, false);
    //        test_gemm_result_mkldnn<X86,X86>(m,n,k,false,false);
    //    }

    if (jit::mayiuse(jit::avx512_core_vnni)) {
        test_gemm_result_mkl_warp<X86, X86>(222, 333, 444, false, false);
    }
    test_gemm_result_mkldnn<X86, X86>(12, 1536 * 4, 512, false, false, PACKED_MKLGEMM);
    test_gemm_result_mkldnn<X86, X86>(12, 1536 * 4, 2048, false, false, PACKED_MKLGEMM);
    test_gemm_result_mkldnn<X86, X86>(4, 1536 * 4, 512, false, false, PACKED_MKLGEMM);
    test_gemm_result_mkldnn<X86, X86>(4, 512, 1536, false, false, PACKED_MKLGEMM);
    test_gemm_result_mkldnn<X86, X86>(1, 1536 * 4, 512, false, false, PACKED_MKLGEMM);
    test_gemm_result_mkldnn<X86, X86>(1, 512, 1536, false, false, PACKED_MKLGEMM);

    //
    //    test_gemm_result<X86, X86>(16,1536*4,512,false,false);
    //    test_gemm_result<X86, X86>(16,1536*4,2048,false,false);
    //    test_gemm_result<X86, X86>(4,1536*4,512,false,false);
    //    test_gemm_result<X86, X86>(512,512,512,false,false);
    //    test_gemm_result<X86, X86>(1024,1024,1024,false,false);

    //    test_gemm_result<X86, X86>(4,512,1536,false,false);
    //    test_gemm_result<X86, X86>(1,1536*4,512,false,false);
    //    test_gemm_result<X86, X86>(1,512,1536,false,false);
    //    test_gemm_result<X86, X86>(256,24*24,128,false,false);
    //    test_gemm_result<X86, X86>(256,24*24,128,false,false);


    //    test_mkl_gemm(32,32,32);
    //    test_mkl_gemm(64,64,64);
    //    test_mkl_gemm(128,128,128);
    //    test_mkl_gemm(256,256,256);
    //    test_mkl_gemm(512,512,512);

    //    test_gemm_result<X86, X86>(32,32,32,false,false);
    //    test_gemm_result<X86, X86>(64,64,64,false,false);
    //    test_gemm_result<X86, X86>(128,128,128,false,false);
    //    test_gemm_result<X86, X86>(256,256,256,false,false);
    //    test_gemm_result<X86, X86>(512,512,512,false,false);
    //    test_gemm_result<X86, X86>(2048,2048,2048,false,false);
    //    test_gemm_result<X86, X86>(4096,4096,4096,false,false);
#endif
}

TEST(TestSaberFunc, test_vender_gemv_float) {

    std::vector<int> m_v = {20, 100, 150, 200, 250, 300};
    std::vector<int> n_v = {20, 100, 150, 200, 250, 300};
    std::vector<int> trans_v{false, true};

    for (auto m : m_v)
        for (auto n : n_v)
            for (auto trans : trans_v) {

#ifdef USE_CUDA
                test_gemv_result<NV, NVHX86>(m, n, trans);
#endif

#ifdef USE_X86_PLACE
                //                test_gemv_result<X86, X86>(m, n, trans);
#endif
            }
}

int main(int argc, char* argv[]) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
#endif
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
