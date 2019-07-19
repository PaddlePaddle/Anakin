#include "saber/core/context.h"
#include "saber/funcs/gemm.h"
#include "saber/funcs/timer.h"
#include "saber/core/tensor.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/calibrate.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "conv_func_helper.h"
#include <vector>
#if defined(USE_X86_PLACE)
#include "saber/funcs/impl/x86/mkl_gemm.h"
#include "saber/funcs/impl/x86/intrinsic_gemm.h"
#include "saber/funcs/impl/x86/intrinsic_packed_fc.h"
#include <emmintrin.h>
#define CLEAR_CACHE 1
#endif
using namespace anakin::saber;
#if defined(USE_X86_PLACE)
const size_t g_cache_size = 10 * 1000 * 1000;
char g_cache[g_cache_size];
void clear_cache(){
    for (int i = 0;i < g_cache_size;i += 64){
        g_cache[i]++;
    }
}
void flush_tensor_cache_out(Tensor<X86>& tensor){
    char* ptr = static_cast<char*>(tensor.data());
    size_t amount=tensor.valid_size() * tensor.get_dtype_size();
    for (size_t i = 0;i < amount;i += 32){
        _mm_clflush(ptr + i);
    }
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

template <typename AType,typename BType,typename CType>
void gemm_check_int8(const int m, const int n, const int k,
                const AType* a, const BType* b, CType* c,
                const float alpha, const float beta,
                const bool trans_a, const bool trans_b,bool is_base_gemm=false) {
    if(is_base_gemm){
//        LOG(INFO)<<"in";
        int lda = k;
        int ldb = k;
        int ldc = n;
        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;
                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += static_cast<CType>(alpha * (int)a[m_i * lda + k_i] * (int)b[n_i * ldb + k_i]);
                }
            }
        }
        return;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int32_t old_c = (beta == 0) ? 0 : c[i * ldc + j];
                int32_t res = 0;
                c[i * ldc + j]*=beta;
                for (int d = 0; d < k; ++d) {
                    res += a[i * lda + d] * b[j * ldb + d];
                }
                c[i * ldc + j] += res * alpha;
            }
        }
        return;
    }
    if (!trans_a && !trans_b) {
        int lda = k;
        int ldb = n;
        int ldc = n;
        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;
                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += static_cast<CType>(alpha * (int)a[m_i * lda + k_i] * (int)b[k_i * ldb + n_i]);
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
                    c[m_i * ldc + n_i] += static_cast<CType>(alpha * a[m_i * lda + k_i] * b[n_i * ldb + k_i]);
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
                    c[m_i * ldc + n_i] += static_cast<CType>(alpha * a[k_i * lda + m_i] * b[k_i * ldb + n_i]);
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
                    c[m_i * ldc + n_i] += static_cast<CType>(alpha * a[k_i * lda + m_i] * b[n_i * ldb + k_i]);
                }
            }
        }
    }
}
template<>
void gemm_check_int8<float,float,float>(const int m, const int n, const int k,
                     const float* a, const float* b, float* c,
                     const float alpha, const float beta,
                     const bool trans_a, const bool trans_b,bool is_base_gemm){
    gemm_check(m,n,k,a,b,c,alpha,beta,trans_a,trans_b);
}
template <typename dtype>
int count_diff(const dtype* src1, const dtype* src2, int size, double max_ratio) {
    if (max_ratio <= 0) {
        max_ratio = 0.1;
    }
    int count = 0;
    for (int i = 0; i < size; ++i) {
        double ratio = fabs(src1[i] - src2[i]) / fabs(src1[i] + src2[i] + 1e-12);
        if (ratio > max_ratio) {
            ++count;
        }
    }
    return count;
}

template <typename TargetType, typename TargetType_H>
void test_gemm_int8_result (int m, int n, int k, bool trans_a, bool trans_b) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;

    Context<TargetType> ctx1(0, 1, 0);
    int generate_arch = Env<NV>::cur_env()[ctx1.get_device_id()]._info._generate_arch;
    // only support 61 arch for now.
    bool arch_check = (generate_arch == 61);
    if (!arch_check) {
                LOG(INFO) << "device not support int8 op!!";
        return;
    }
    Gemm<TargetType, VENDER_IMPL, char, float> gemm_vender;
    Gemm<TargetType, SABER_IMPL, char, float> gemm_saber;
    SaberStatus vender_status = gemm_vender.init(trans_a, trans_b, m, n, k, ctx1);
    SaberStatus saber_status = gemm_saber.init(trans_a, trans_b, m, n, k, ctx1);

    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({m, k}, Layout_HW);
    Shape b_shape({k, n}, Layout_HW);
    Shape c_shape({m, n}, Layout_HW);

    Tensor<TargetType> a_dev_int8, b_dev_int8;
    Tensor<TargetType> a_scale, b_scale;
    Shape a_scale_shape({m}, Layout_W);
    Shape b_scale_shape({n}, Layout_W);
    a_dev_int8.re_alloc(a_shape, AK_INT8);
    b_dev_int8.re_alloc(b_shape, AK_INT8);
    a_scale.re_alloc(a_scale_shape, AK_FLOAT);
    b_scale.re_alloc(b_scale_shape, AK_FLOAT);

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

    int ts = 100;
    if (vender_status == SaberSuccess) {

        /// step 1: calibrate matrix a into int8 matrix
        ///         using row direction.
        /// input : const float* src
        /// output: char* dst_int8
        ///         float* scale
        float2char(trans_a, (signed char*)a_dev_int8.mutable_data(),
                       (const float*)a_dev.data(),
                       (float*)a_scale.mutable_data(),
                       trans_a?k:m, trans_a?m:k, ctx1);

        /// step 2: calibrate matrix a into int8 matrix
        ///         using col direction.
        /// input : const float* src
        /// output: char* dst_int8
        ///         float* scale
        float2char(!trans_b, (signed char*)b_dev_int8.mutable_data(),
                       (const float*)b_dev.data(),
                       (float*)b_scale.mutable_data(),
                       trans_b?n:k, trans_b?k:n, ctx1);

        /// step 3: dispatch matrix multiply using int8 gemm
        gemm_vender.dispatch(alpha, beta,
                             (const char *) a_dev_int8.data(),
                             (const char *) b_dev_int8.data(),
                             (float *) c_dev.mutable_data());

        /// step 4: convert int32 into float32 using fix2float.
        /// input : float* dst
        ///         const scaleA(row scale)
        ///         const scaleB(col scale)
        /// output: float* dst
        fix2float((float*)c_dev.mutable_data(),
                  (const float*)a_scale.data(),
                  (const float*)b_scale.data(),
                  alpha, beta, m, n, ctx1);

        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();

        c_host.copy_from(c_dev);
        gemm_check(m, n, k, (const float *) a_host.data(), (const float *) b_host.data(),
                   (float *) c_check.mutable_data(),
                   alpha, beta, trans_a, trans_b);

        int counts = count_diff((const float*)c_check.data(), (const float*)c_host.data(),
                                c_check.valid_size(), 1e-1);

        if (((double)counts / (double)c_host.valid_size()) > 0.05) {
            print_tensor_valid(c_check);
            print_tensor_valid(c_host);
            LOG(FATAL) << "VENDER: FAIL!!!! counts = " <<counts
                       << "m = "<< m << " n = "<< n << " k = "<< k;
        }
        for (int t = 0; t < ts; ++t) {
            vender_time.start(ctx1);
            gemm_vender.dispatch(alpha, beta,
                                 (const char *) a_dev_int8.data(),
                                 (const char *) b_dev_int8.data(),
                                 (float *) c_dev.mutable_data());
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            vender_time.end(ctx1);
        }
    }
    if (saber_status == SaberSuccess) {

        /// step 1: calibrate matrix a into int8 matrix
        ///         using row direction.
        /// input : const float* src
        /// output: char* dst_int8
        ///         float* scale
        float2char(trans_a, (signed char*)a_dev_int8.mutable_data(),
                   (const float*)a_dev.data(),
                   (float*)a_scale.mutable_data(),
                   trans_a?k:m, trans_a?m:k, ctx1);

        /// step 2: calibrate matrix a into int8 matrix
        ///         using col direction.
        /// input : const float* src
        /// output: char* dst_int8
        ///         float* scale
        float2char(!trans_b, (signed char*)b_dev_int8.mutable_data(),
                   (const float*)b_dev.data(),
                   (float*)b_scale.mutable_data(),
                   trans_b?n:k, trans_b?k:n, ctx1);

        /// step 3: dispatch matrix multiply using int8 gemm
        gemm_saber.dispatch(alpha, beta,
                            (const char *) a_dev_int8.data(),
                            (const char *) b_dev_int8.data(),
                            (float *) c_dev.mutable_data());

        /// step 4: convert int32 into float32 using fix2float.
        /// input : float* dst
        ///         const scaleA(row scale)
        ///         const scaleB(col scale)
        /// output: float* dst
        fix2float((float*)c_dev.mutable_data(),
                  (const float*)a_scale.data(),
                  (const float*)b_scale.data(),
                  alpha, beta, m, n, ctx1);

        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();

        c_host.copy_from(c_dev);
        gemm_check(m, n, k, (const float *) a_host.data(), (const float *) b_host.data(),
                   (float *) c_check.mutable_data(),
                   alpha, beta, trans_a, trans_b);

        int counts = count_diff((const float*)c_check.data(), (const float*)c_host.data(),
                                c_check.valid_size(), 1e-1);

        if (((double)counts / (double)c_host.valid_size()) > 0.05) {
            print_tensor_valid(c_check);
            print_tensor_valid(c_host);
                    LOG(FATAL) << "VENDER: FAIL!!!! counts = " <<counts
                               << "m = "<< m << " n = "<< n << " k = "<< k;
        }
        for (int t = 0; t < ts; ++t) {
            saber_time.start(ctx1);
            gemm_saber.dispatch(alpha, beta,
                                 (const char *) a_dev_int8.data(),
                                 (const char *) b_dev_int8.data(),
                                 (float *) c_dev.mutable_data());
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
#if defined(USE_X86_PLACE)
template <typename TargetType, typename TargetType_H,DataType AK_AType,DataType AK_BType>
void test_gemm_result_mkldnn(int m, int n, int k, bool trans_a, bool trans_b, bool packed_gemm = false) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;
    typedef typename DataTrait<TargetType,AK_AType>::Dtype AType;
    typedef typename DataTrait<TargetType,AK_BType>::Dtype BType;
    Context<TargetType> ctx1(0, 1, 0);
    MklDnnGemm<AType , BType, int> gemm_vender;


    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({m, k}, Layout_HW);
    Shape b_shape({k, n}, Layout_HW);
    Shape c_shape({m, n}, Layout_HW);

    a_dev.re_alloc(a_shape, AK_AType);
    b_dev.re_alloc(b_shape, AK_BType);
    c_dev.re_alloc(c_shape, AK_INT32);

    a_host.re_alloc(a_shape, AK_AType);
    b_host.re_alloc(b_shape, AK_BType);
    c_host.re_alloc(c_shape, AK_INT32);
    c_check.re_alloc(c_shape, AK_INT32);
    if (AK_AType==AK_UINT8){
        fill_tensor_rand(a_dev, 0.f, 240.f);
        fill_tensor_rand(b_dev, -150.f, 150.f);
    }else if(AK_AType==AK_INT8){
        fill_tensor_rand(a_dev, -126.f, 126.f);
        fill_tensor_rand(b_dev, -126.f, 126.f);
    }else{
        fill_tensor_rand(a_dev, -126.f, 126.f);
        fill_tensor_rand(b_dev, -126.f, 126.f);
    }

    a_host.copy_from(a_dev);
    b_host.copy_from(b_dev);

    SaberStatus vender_status =SaberSuccess;
    if(packed_gemm) {
        vender_status = gemm_vender.init(trans_a, trans_b, m, n, k, ctx1, (BType *) b_dev.data(),PACKED_MKLGEMM);
    }else{
        vender_status = gemm_vender.init(trans_a, trans_b, m, n, k, ctx1, (BType *) b_dev.data(),NORMAL_MKLGEMM);
        fill_tensor_rand(b_dev, -150.f, 150.f);
        b_host.copy_from(b_dev);
    }

    SaberTimer<TargetType> vender_time, saber_time;
    int ts = 200;

    if (vender_status == SaberSuccess) {
        gemm_vender.dispatch(alpha, beta,m,
                             (const AType*) a_dev.data(),
                             (const BType*) b_dev.data(),
                             (int*) c_dev.mutable_data());
        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        c_host.copy_from(c_dev);

        gemm_check_int8(m, n, k, (const AType*) a_host.data(), (const BType*) b_host.data(),
                   (int*) c_check.mutable_data(),
                   alpha, beta, trans_a, trans_b);
        double max_ratio = 0.f, max_diff = 0.f;
        tensor_cmp_host_mlu((const int*) c_check.data(), (const int*) c_host.data(),
                            c_check.valid_size(), max_ratio, max_diff);

        if (max_ratio > 1e-3) {
            LOG(FATAL) << "VENDER: FAIL!!!! max_ratio = " << max_ratio << " max_diff: " << max_diff
                               << "m = " << m << " n = " << n << " k = " << k;
        }

        for (int t = 0; t < ts; ++t) {
#if CLEAR_CACHE
            flush_tensor_cache_out(a_dev);
            flush_tensor_cache_out(b_dev);
            flush_tensor_cache_out(c_dev);
#endif
            vender_time.start(ctx1);
            gemm_vender.dispatch(alpha, beta,m,
                                 (const AType*) a_dev.data(),
                                 (const BType*) b_dev.data(),
                                 (int*) c_dev.mutable_data());
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            vender_time.end(ctx1);
        }
    }else{
        LOG(ERROR)<<"MklDnnGemm not impl";
    }

    double work = (double)m * n * k * 2;
    double vender_time_ms = (vender_status == SaberSuccess ? vender_time.get_average_ms() : 1e10);
    double vender_speed = work / vender_time_ms / 1000.0 / 1000.0;
    LOG(INFO)<<"mkldnn " <<m<<","<<n<<","<<k<<"::"<< "gops " << vender_speed;
//    LOG(INFO) << "Vender time: " << (vender_status == SaberSuccess ? vender_time.get_average_ms() : 0)
//              << "ms ,speed = " << vender_speed << "gfloat/s";
}

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

template <typename TargetType, typename TargetType_H,DataType AK_AType,DataType AK_BType,DataType AK_CType>
void test_gemm_result_intrin_me(int m, int n, int k, bool trans_a, bool trans_b,bool check_correct=true,PackedFCAlg alg=DotReduction) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;
    typedef typename MyDataTrait<AK_AType>::Dtype AType;
    typedef typename MyDataTrait<AK_BType>::Dtype BType;
    typedef typename MyDataTrait<AK_CType>::Dtype CType;
    Context<TargetType> ctx1(0, 1, 0);
    PackedFC<AK_AType,AK_BType,AK_CType> gemm_vender;


    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({1,1,m, k}, Layout_NCHW);
    Shape b_shape({1,1,k, n}, Layout_NCHW);
    Shape c_shape({1,1,m, n}, Layout_NCHW);

    a_dev.re_alloc(a_shape, AK_AType);
    b_dev.re_alloc(b_shape, AK_BType);
    c_dev.re_alloc(c_shape, AK_CType);

    a_host.re_alloc(a_shape, AK_AType);
    b_host.re_alloc(b_shape, AK_BType);
    c_host.re_alloc(c_shape, AK_CType);
    c_check.re_alloc(c_shape, AK_CType);
    if(AK_AType==AK_UINT8){
        fill_tensor_rand(a_dev, 0.f, 220.f);

    }else if(AK_AType==AK_FLOAT){
        fill_tensor_rand(a_dev, -1.f, 1.f);
        a_dev.set_scale({1.f/127.f});
    } else{
//        fill_tensor_const(a_dev,1);
        fill_tensor_rand(a_dev);

    }

    if(AK_BType==AK_INT8){
//        fill_tensor_const(b_dev,1);
        fill_tensor_rand(b_dev);
    }else if(AK_BType==AK_FLOAT){
        fill_tensor_rand(b_dev,-1.f,1.f);
        b_dev.set_scale({1.f/127.f});
    }else{
        LOG(FATAL)<<"not impl";
    }

    if(AK_CType==AK_FLOAT){
        c_dev.set_scale({1.f});
    }

    a_host.copy_from(a_dev);
    b_host.copy_from(b_dev);


    SaberStatus vender_status = SaberNotInitialized;
    if(AK_CType==AK_FLOAT){
        CHECK_EQ(a_dev.get_scale().size(),1);
        CHECK_EQ(c_dev.get_scale().size(),1);
        vender_status=gemm_vender.init(n,k,b_dev,a_dev.get_scale()[0],c_dev.get_scale()[0],alg);

    }else{
        vender_status=gemm_vender.init(n,k,b_dev,1.f,1.f,alg);
    }



    if (vender_status == SaberSuccess) {

//        LOG(INFO)<<"m = "<<m<<","<<n<<","<<k;
        gemm_vender.dispatch(m,n,k,
                              a_dev,
                             c_dev);
        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        c_host.copy_from(c_dev);
        gemm_check_int8(m, n, k, (const AType*) a_host.data(), (const BType*) b_host.data(),
                        (CType*) c_check.mutable_data(),
                        alpha, beta, trans_a, trans_b);
        double max_ratio = 0.f, max_diff = 0.f;
        double mlu_diff=0.f;
//        tensor_cmp_host((const CType*) c_check.data(), (const CType*) c_host.data(),
//                        c_check.valid_size(), max_ratio, max_diff);

        tensor_cmp_host((const CType*) c_check.data(), (const CType*) c_host.data(),
                        c_check.valid_size(), mlu_diff);
//        LOG(INFO)<<"mludiff = "<<mlu_diff;

//        print_tensor(a_dev);
//        print_tensor(b_dev);
//        print_tensor(c_dev);
//        LOG(INFO)<<"max ratio "<<max_ratio;

        if(check_correct) {
            if (mlu_diff > 1e-2) {
//            print_tensor(a_dev);
//            print_tensor(b_dev);
                print_tensor_valid(c_check);
                print_tensor_valid(c_host);
                        LOG(FATAL) << "VENDER: FAIL!!!! max_ratio = " << max_ratio << " max_diff: " << max_diff
                                   << "m = " << m << " n = " << n << " k = " << k;
            }
//            LOG(INFO)<<"passed";
        }

    }else{
        LOG(ERROR)<<"MklDnnGemm not impl";
    }

    SaberTimer<TargetType> vender_time, saber_time;
    int ts = 300;
    int warm_up=0;

    for (int t = 0; t < warm_up; ++t) {
        gemm_vender.dispatch(m,n,k,
                              a_dev,
                             c_dev);
    }
    for (int t = 0; t < ts; ++t) {
#if CLEAR_CACHE
        flush_tensor_cache_out(a_dev);
        flush_tensor_cache_out(b_dev);
        flush_tensor_cache_out(c_dev);
        flush_tensor_cache_out((gemm_vender._inner_weights));
#endif
        vender_time.start(ctx1);
        gemm_vender.dispatch(m,n,k,
                             a_dev,
                             c_dev);
        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        vender_time.end(ctx1);
    }

    double work = (double)m * n * k * 2;
    double vender_time_ms = (vender_status == SaberSuccess ? vender_time.get_average_ms() : 1e10);
    double vender_speed = work / vender_time_ms / 1000.0 / 1000.0;
    LOG(INFO)<<"me " <<m<<","<<n<<","<<k<<"::"<< "gops " << vender_speed;
//    LOG(INFO) << "Vender time: " << (vender_status == SaberSuccess ? vender_time.get_average_ms() : 0)
//              << "ms ,speed = " << vender_speed << "gfloat/s";
}

template <typename TargetType, typename TargetType_H,DataType AK_AType,DataType AK_BType>
void test_gemm_result_intrin(int m, int n, int k, bool trans_a, bool trans_b,bool is_base_gemm=false) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;
    typedef typename DataTrait<TargetType,AK_AType>::Dtype AType;
    typedef typename DataTrait<TargetType,AK_BType>::Dtype BType;
    Context<TargetType> ctx1(0, 1, 0);
    IntrinsicGemm<AType , BType, int> gemm_vender;
    SaberStatus vender_status = gemm_vender.init(trans_a, trans_b, m, n, k, ctx1);

    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({m, k}, Layout_HW);
    Shape b_shape({k, n}, Layout_HW);
    Shape c_shape({m, n}, Layout_HW);

    a_dev.re_alloc(a_shape, AK_AType);
    b_dev.re_alloc(b_shape, AK_BType);
    c_dev.re_alloc(c_shape, AK_INT32);

    a_host.re_alloc(a_shape, AK_AType);
    b_host.re_alloc(b_shape, AK_BType);
    c_host.re_alloc(c_shape, AK_INT32);
    c_check.re_alloc(c_shape, AK_INT32);
    if(AK_AType==AK_UINT8){
//        fill_tensor_rand(a_dev, 0.f, 250.f);
//        fill_tensor_rand(b_dev, -126.f, 126.f);
        fill_tensor_rand(a_dev, 0.f, 220.f);
        fill_tensor_rand(b_dev, -150.f, 150.f);
    }else{
        fill_tensor_rand(a_dev);
        fill_tensor_rand(b_dev);
    }

    a_host.copy_from(a_dev);
    b_host.copy_from(b_dev);
    SaberTimer<TargetType> vender_time, saber_time;
    int ts = 1000;
    int warm_up = 100;
//    LOG(INFO)<<"vender_status "<<vender_status<<",is_base_gemm = "<<is_base_gemm;

    if (vender_status == SaberSuccess) {
        gemm_vender.dispatch(alpha, beta,
                             (const AType*) a_dev.data(),
                             (const BType*) b_dev.data(),
                             (int*) c_dev.mutable_data());
        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();
        c_host.copy_from(c_dev);
        gemm_check_int8(m, n, k, (const AType*) a_host.data(), (const BType*) b_host.data(),
                        (int*) c_check.mutable_data(),
                        alpha, beta, trans_a, trans_b,is_base_gemm);
        double max_ratio = 0.f, max_diff = 0.f;
        tensor_cmp_host((const int*) c_check.data(), (const int*) c_host.data(),
                        c_check.valid_size(), max_ratio, max_diff);

        if (max_ratio > 1e-3) {

                    LOG(FATAL) << "VENDER: FAIL!!!! max_ratio = " << max_ratio << " max_diff: " << max_diff
                               << "m = " << m << " n = " << n << " k = " << k;
        }
        for (int t = 0; t < warm_up; ++t) {
            gemm_vender.dispatch(alpha, beta,
                                 (const AType*) a_dev.data(),
                                 (const BType*) b_dev.data(),
                                 (int*) c_dev.mutable_data());
        }

        for (int t = 0; t < ts; ++t) {
            vender_time.start(ctx1);
            gemm_vender.dispatch(alpha, beta,
                                 (const AType*) a_dev.data(),
                                 (const BType*) b_dev.data(),
                                 (int*) c_dev.mutable_data());
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            vender_time.end(ctx1);
        }
    }else{
                LOG(ERROR)<<"MklDnnGemm not impl";
    }

    double work = m * n * k * 2;
    double vender_time_ms = (vender_status == SaberSuccess ? vender_time.get_average_ms() : 1e10);
    double vender_speed = work / vender_time_ms / 1000.0 / 1000.0;
    LOG(INFO)<<"audio "<<m<<","<<n<<","<<k<<"::"<< "gops " << vender_speed;
//    LOG(INFO) << "Vender time: " << (vender_status == SaberSuccess ? vender_time.get_average_ms() : 0)
//              << "ms ,speed = " << vender_speed << "gfloat/s";
}

#endif
TEST(TestSaberFunc, test_vender_gemm_float) {

    srand(12345);
    std::vector<int> m_v = {40, 20, 140, 200, 300};
    std::vector<int> n_v = {10, 20, 140, 200, 300};
    std::vector<int> k_v = {40, 20, 140, 200, 300};
    std::vector<int> trans_a_v{false};
    std::vector<int> trans_b_v{false};

    for (auto m : m_v)
    for (auto n : n_v)
    for (auto k : k_v)
    for (auto trans_a : trans_a_v)
    for (auto trans_b : trans_b_v) {

#ifdef USE_CUDA
        test_gemm_int8_result<NV, NVHX86>(m, n, k, trans_a, trans_b);
#endif
    }
#if defined(USE_X86_PLACE)
#if 1//defined(__AVX2__)
    //    test_gemm_result_intrin<X86,X86,AK_INT8,AK_INT8>(12,1536*4,512,false,false,true);
    //    test_gemm_result_intrin<X86,X86,AK_INT8,AK_INT8>(12,1536*4,2048,false,false,true);
    //    test_gemm_result_intrin<X86,X86,AK_INT8,AK_INT8>(4,1536*4,512,false,false,true);
    //    test_gemm_result_intrin<X86,X86,AK_INT8,AK_INT8>(4,512,1536,false,false,true);
    //    test_gemm_result_intrin<X86,X86,AK_INT8,AK_INT8>(1,1536*4,512,false,false,true);
    //    test_gemm_result_intrin<X86,X86,AK_INT8,AK_INT8>(1,512,1536,false,false,true);


    //    test_gemm_result_intrin_me<X86,X86,AK_FLOAT,AK_FLOAT,AK_FLOAT>(4,4,32,false,false);
    //    test_gemm_result_intrin_me<X86,X86,AK_FLOAT,AK_FLOAT,AK_FLOAT>(16,1536*4,512,false,false);
    //    test_gemm_result_intrin_me<X86,X86,AK_FLOAT,AK_FLOAT,AK_FLOAT>(16,1536*4,2048,false,false);


//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(16,1536*4,512,false,false,true,DotSplitK);


//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(16,1536*4,512,false,false,true,DotReductionPacked);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(16,1536*4,2048,false,false,true,DotReductionPacked);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(4,1536*4,512,false,false,true,DotReductionPacked);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(512,512,512,false,false,true,DotReductionPacked);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(1024,1024,1024,false,false,true,DotReductionPacked);

//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(16,1536*4,512,false,false,true,DotReduction);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(16,1536*4,2048,false,false,true,DotReduction);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(4,1536*4,512,false,false,true,DotReduction);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(512,512,512,false,false,true,DotReduction);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(1024,1024,1024,false,false,true,DotReduction);

//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(16,1536*4,512,false,false,true,DotAdd);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(16,1536*4,2048,false,false,true,DotAdd);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(4,1536*4,512,false,false,true,DotAdd);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(512,512,512,false,false,true,DotAdd);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(1024,1024,1024,false,false,true,DotAdd);

//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(16,1536*4,512,false,false,true,DotSplitK);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(16,1536*4,2048,false,false,true,DotSplitK);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(4,1536*4,512,false,false,true,DotSplitK);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(512,512,512,false,false,true,DotSplitK);
//        test_gemm_result_intrin_me<X86,X86,AK_INT8,AK_INT8,AK_INT32>(1024,1024,1024,false,false,true,DotSplitK);
//    test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(16, 16, 16, false, false);
#endif

    if (jit::mayiuse(jit::avx512_core_vnni)) {
        for (auto m : {
                1, 3, 6, 16
        }) {
            for (auto n : {
                    4, 12, 17, 23
            }) {
                for (auto k : {
                        3, 12, 16, 32, 33
                }) {
                    test_gemm_result_mkldnn<X86, X86, AK_UINT8, AK_INT8>(m, n, k, false, false, true);
                    test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(m, n, k, false, false, true);
                }
            }
        }
//        test_gemm_result_mkldnn<X86, X86, AK_UINT8, AK_INT8>(4, 4, 4, false, false, true);
//        test_gemm_result_mkldnn<X86, X86, AK_UINT8, AK_INT8>(2, 3, 4, false, false, true);
//        test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(2, 3, 4, false, false, true);
//        test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(2, 3, 4, false, false, false);

//        test_gemm_result_mkldnn<X86, X86, AK_UINT8, AK_INT8>(16, 1536 * 4, 512, false, false, true);
//        test_gemm_result_mkldnn<X86, X86, AK_UINT8, AK_INT8>(16, 1536 * 4, 512, false, false, false);
//        test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(16, 1536 * 4, 512, false, false, true);
//        test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(16, 1536 * 4, 512, false, false, false);

//        test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(16, 1536 * 4, 2048, false, false);
//        test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(4, 1536 * 4, 512, false, false);
//        test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(512, 512, 512, false, false);
//        test_gemm_result_mkldnn<X86, X86, AK_INT8, AK_INT8>(1024, 1024, 1024, false, false);
    }

#endif

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
