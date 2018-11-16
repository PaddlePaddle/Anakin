#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SASS_FUNCS_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SASS_FUNCS_H

#include "saber_types.h"
#include <stdlib.h>
#include <stdio.h>
#include <functional>
#include <map>

void invoke_test();

void invoke_test_2();

namespace anakin {
namespace saber {

//Round a / b to nearest higher integer value
inline int i_div_up(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int i_align_up(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

inline int bin(int var){
    int x = (var >= 0) ? var : -var;
    int bits;
    for (bits = 0; x != 0; ++bits){
        x >>= 1;
    }
    return bits;
}

inline std::pair<unsigned int, unsigned int> 
magic_32_div(long long int nmax, int div)
{
    unsigned m = -1;
    unsigned int p;
    long long int nc = ((nmax + 1) / div) * div - 1;
    int nbits = bin(nmax);
    int range = 2 * nbits + 1;
    for (p = 0; p < range; p++){
        long long int exp = 1 << p;
        long long int mod = div - 1 - (exp - 1) % div;
        if (exp > nc * mod)
        {
            m = (unsigned) ((exp + mod) / div);
            return std::make_pair(m, p);
        }
    }
    return std::make_pair(-1, -1);
}

template <typename DataType, typename OpType>
void winograd_conv(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);   

template <typename DataType, typename OpType>
void winograd_conv_relu(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);

template <typename DataType, typename OpType>
void winograd_conv_relu_pooling(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);

template <typename DataType, typename OpType>
void winograd_conv_eltwise(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta,
    EltwiseType elt_type,
    cudaStream_t cuda_stream);

template <typename DataType, typename OpType>
void direct_conv_Kdivis4(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);  

template <typename DataType, typename OpType>
void direct_conv_Kindiv4(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);  

template <typename DataType, typename OpType>
void direct_conv_bias_Kdivis4(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);   

template <typename DataType, typename OpType>
void direct_conv_bias_Kindiv4(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);   

template <typename DataType, typename OpType>
void direct_conv_bias_relu_Kdivis4(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);   

template <typename DataType, typename OpType>
void direct_conv_bias_relu_Kindiv4(const DataType* src,
    DataType* dst,
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);   


template <typename DataType, typename OpType>
void direct_conv_bias_relu_maxpool2k2s0p_Kdivis4(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);   

template <typename DataType, typename OpType>
void direct_conv_bias_relu_maxpool2k2s0p_Kindiv4(const DataType* src,
    DataType* dst, 
    const OpType* weight,
    const DataType* bias,
    int img_num,
    int img_in_channel,
    int img_in_height,
    int img_in_width,
    int img_out_channel,
    int img_out_height,
    int img_out_width,
    int img_in_channel_stride,
    int img_in_height_stride,
    int img_in_width_stride,
    int img_out_channel_stride,
    int img_out_height_stride,
    int img_out_width_stride,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int group,
    float alpha,
    float beta, 
    cudaStream_t cuda_stream);

void ker_deconv_implicit_gemm_k4_s2_p1_16x64(
        float* dout, const float *din,
        const float* weights, const float* bias,
        int num, int hin, int win, int hout, int wout,
        int ch_in, int ch_out, cudaStream_t &stream);

void ker_deconv_implicit_gemm_k4_s2_p1_32x32_relu(
        float* dout, const float *din,
        const float* weights, const float* bias,
        int num, int hin, int win, int hout, int wout,
        int ch_in, int ch_out, cudaStream_t &stream);

__inline__
bool ifVec(int m, int n, int k,
           int lda, int ldb, int ldc)
{
    bool vec_a = false;
    bool vec_b = false;
    bool vec_c = false;

    vec_a = ((lda & 3) == 0) && ((k & 3) == 0);
    vec_b = ((ldb & 3) == 0) && ((n & 3) == 0);
    vec_c = ((ldc & 3) == 0) && ((n & 3) == 0);

    return vec_a && vec_b && vec_c;
}

void ker_gemm_32x32x32_NN_bias_relu(const int M, const int N, const int K,
                                    const float alpha, const float* A,
                                    const float beta, const float* B,
                                    float* C, const float* bias, cudaStream_t cuda_stream);

void ker_gemm_32x32x32_NN_vec_bias_relu(const int M, const int N, const int K,
                                        const float alpha, const float* A,
                                        const float beta, const float* B,
                                        float* C, const float* bias, cudaStream_t cuda_stream);

void ker_gemm_32x32x32_NN_bias(const int M, const int N, const int K,
                               const float alpha, const float* A,
                               const float beta, const float* B,
                               float* C, const float* bias, cudaStream_t cuda_stream);

void ker_gemm_32x32x32_NN_vec_bias(const int M, const int N, const int K,
                                   const float alpha, const float* A,
                                   const float beta, const float* B,
                                   float* C, const float* bias, cudaStream_t cuda_stream);

template <int tile>
void ker_sgemm_nn(const int M, const int N, const int K,
                  const int lda, const int ldb, const int ldc,
                  const float alpha, const float* A,
                  const float beta, const float* B,
                  float* C, cudaStream_t cuda_stream);
template <int tile>
void ker_sgemm_nt(const int M, const int N, const int K,
                  const int lda, const int ldb, const int ldc,
                  const float alpha, const float* A,
                  const float beta, const float* B,
                  float* C, cudaStream_t cuda_stream);
template <int tile>
void ker_sgemm_tn(const int M, const int N, const int K,
                  const int lda, const int ldb, const int ldc,
                  const float alpha, const float* A,
                  const float beta, const float* B,
                  float* C, cudaStream_t cuda_stream);
template <int tile>
void ker_sgemm_tt(const int M, const int N, const int K,
                  const int lda, const int ldb, const int ldc,
                  const float alpha, const float* A,
                  const float beta, const float* B,
                  float* C, cudaStream_t cuda_stream);
template <int tile>
void ker_sgemm_nn_vec(const int M, const int N, const int K,
                      const int lda, const int ldb, const int ldc,
                      const float alpha, const float* A,
                      const float beta, const float* B,
                      float* C, cudaStream_t cuda_stream);
template <int tile>
void ker_sgemm_nt_vec(const int M, const int N, const int K,
                      const int lda, const int ldb, const int ldc,
                      const float alpha, const float* A,
                      const float beta, const float* B,
                      float* C, cudaStream_t cuda_stream);
template <int tile>
void ker_sgemm_tn_vec(const int M, const int N, const int K,
                      const int lda, const int ldb, const int ldc,
                      const float alpha, const float* A,
                      const float beta, const float* B,
                      float* C, cudaStream_t cuda_stream);
template <int tile>
void ker_sgemm_tt_vec(const int M, const int N, const int K,
                      const int lda, const int ldb, const int ldc,
                      const float alpha, const float* A,
                      const float beta, const float* B,
                      float* C, cudaStream_t cuda_stream);

template <bool TransA, bool TransB, int tile>
void ker_sgemm_sass(const int M, const int N, const int K,
                    const float alpha, const float* A,
                    const float beta, const float* B,
                    float* C, cudaStream_t cuda_stream);

std::function<void(const int, const int, const int,
                   const float, const float*, const float,
                   const float*, float*, cudaStream_t)>
saber_find_fast_sass_gemm(const bool TransA, const bool TransB,
                          const int M, const int N, const int K);

template <bool with_relu>
void conv_gemm_k1s1p0(int num, int in_stride, int out_stride,
                      float* out, const float* img,
                      const float* weights, int out_channel,
                      int in_channel, int img_h, int img_w,
                      const float* bias, cudaStream_t cuda_stream,
                      float a = 1.f, float b = 0.f) {

    float alpha = a;
    float beta = b;
    int m = out_channel;
    int k = in_channel;
    int n = img_h * img_w;
    if (ifVec(m, n, k, k, n, n)) {
        if (with_relu) {
            for (int i = 0; i < num; ++i) {
                ker_gemm_32x32x32_NN_vec_bias_relu(m, n, k,
                                                   alpha, weights,
                                                   beta, img + i * in_stride,
                                                   out + i * out_stride, bias,
                                                   cuda_stream);
            }
        } else {
            for (int i = 0; i < num; ++i) {
                ker_gemm_32x32x32_NN_vec_bias(m, n, k,
                                              alpha, weights,
                                              beta, img + i * in_stride,
                                              out + i * out_stride, bias,
                                              cuda_stream);
            }
        }
    } else {
        if (with_relu) {
            for (int i = 0; i < num; ++i) {
                ker_gemm_32x32x32_NN_bias_relu(m, n, k,
                                               alpha, weights,
                                               beta, img + i * in_stride,
                                               out + i * out_stride, bias,
                                               cuda_stream);
            }
        } else {
            for (int i = 0; i < num; ++i) {
                ker_gemm_32x32x32_NN_bias(m, n, k,
                                          alpha, weights,
                                          beta, img + i * in_stride,
                                          out + i * out_stride, bias,
                                          cuda_stream);
            }
        }
    }
}

} // namespace saber
} // namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SASS_FUNCS_H
