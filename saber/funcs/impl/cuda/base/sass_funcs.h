#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SASS_FUNCS_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SASS_FUNCS_H
#include "saber/saber_types.h"
#include <stdlib.h>
#include <stdio.h>
#include <map>


void invoke_test();

void invoke_test_2();

namespace anakin{
namespace saber{
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

}
}

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SASS_FUNCS_H
