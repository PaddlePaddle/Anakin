//#include "saber/funcs/impl/cuda/saber_conv_act.h"

#include "saber/saber_types.h"
#include "saber/core/common.h"
#include <sm_61_intrinsics.h>
namespace anakin{

namespace saber{

template <bool bias_flag, bool relu_flag>
__global__ void depthwise_conv_1d(const int nthreads,
        const float* const din, const int num, const int channels,
        const int hin, const int win, const int hout,
        const int wout, const int kernel_h, const int kernel_w,
        const int stride_h, const int stride_w, const int pad_h, const int pad_w,
        float* const dout, const float* const weight, const float* const bias) {

    int size_channel_in = hin * win;
    int size_channel_out = hout * wout;
    int size_kernel = kernel_h * kernel_w;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int pw = index % wout;
        const int ph = (index / wout) % hout;
        const int c = (index / size_channel_out) % channels;
        const int n = index / size_channel_out / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = hstart + kernel_h;
        int wend = wstart + kernel_w;

        int khstart = hstart < 0 ? 0 - hstart : 0;
        int kwstart = wstart < 0 ? 0 - wstart : 0;

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, hin);
        wend = min(wend, win);
        float aveval = 0;
        const float* const bottom_slice = din + (n * channels + c) * size_channel_in;
        const float* const weight_slice = weight + c * size_kernel;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                aveval += bottom_slice[h * win + w]
                        * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
            }
        }
        if (bias_flag) {
            aveval+=bias[c];
        }
        if (relu_flag) {
            aveval = max(aveval, (float)0);
        }
        dout[index] = aveval;
    }
}

template <bool relu_flag>
SaberStatus saber_depthwise_conv_act(const float* input, float* output,
    int num, int cin, int hin, int win, int hout, int wout,
    int kw, int kh, int stride_w, int stride_h, int pad_w, int pad_h,
    const float* weights, const float* bias, cudaStream_t stream) {

    const int count = num * cin * hout * wout;
    if (bias != nullptr) {
        depthwise_conv_1d<true, relu_flag><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
                count, input, num, cin, hin, win, hout, wout, kh,
                kw, stride_h, stride_w, pad_h, pad_w,
                output, weights, bias);
    } else {
        depthwise_conv_1d<false, relu_flag><<< CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>> (
                count, input, num, cin, hin, win, hout, wout, kh,
                kw, stride_h, stride_w, pad_h,
                pad_w, output, weights, nullptr);
    }
    return SaberSuccess;
}

#define MASK3 0xff000000
#define MASK2 0x00ff0000
#define MASK1 0x0000ff00
#define MASK0 0x000000ff

template <bool bias_flag, bool relu_flag>
__global__ void depthwise_conv_1d_s8_s8(const int nthreads,
        const void* din, const int num, const int channels,
        const int hin, const int win, const int hout,
        const int wout, const int kernel_h, const int kernel_w,
        const int stride_h, const int stride_w, const int pad_h, const int pad_w,
        void* dout, const void* weight, const float* bias, float alpha = 1.f) {
#if __CUDA_ARCH__ > 600
    int size_channel_in = hin * win;
    int size_channel_out = hout * wout;
    int size_kernel = kernel_h * kernel_w;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int pw = index % wout;
        const int ph = (index / wout) % hout;
        const int c = (index / size_channel_out) % channels;
        const int n = index / size_channel_out / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = hstart + kernel_h;
        int wend = wstart + kernel_w;

        int khstart = hstart < 0 ? 0 - hstart : 0;
        int kwstart = wstart < 0 ? 0 - wstart : 0;

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, hin);
        wend = min(wend, win);

        int aveval0 = 0;
        int aveval1 = 0;
        int aveval2 = 0;
        int aveval3 = 0;

        const int* bottom_slice = ((const int*)din);
        bottom_slice += (n * channels + c) * size_channel_in;
        const int* weight_slice= (const int*)weight;
        weight_slice += c * size_kernel;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int in_data = bottom_slice[h * win + w];
                int weight_data = weight_slice[(khstart + h - hstart) * kernel_w
                                          + (kwstart + w - wstart)];

                int mask_weight;
                mask_weight = MASK0 & weight_data;
                aveval0 = __dp4a(in_data, mask_weight, aveval0);
                mask_weight = MASK1 & weight_data;
                aveval1 = __dp4a(in_data, mask_weight, aveval1);
                mask_weight = MASK2 & weight_data;
                aveval2 = __dp4a(in_data, mask_weight, aveval2);
                mask_weight = MASK3 & weight_data;
                aveval3 = __dp4a(in_data, mask_weight, aveval3);
            }
        }
        float fa0 = static_cast<float>(aveval0);
        float fa1 = static_cast<float>(aveval1);
        float fa2 = static_cast<float>(aveval2);
        float fa3 = static_cast<float>(aveval3);
        fa0 *= alpha;
        fa1 *= alpha;
        fa2 *= alpha;
        fa3 *= alpha;
        if (bias_flag) {
            fa0 += bias[4 * c + 0];
            fa1 += bias[4 * c + 1];
            fa2 += bias[4 * c + 2];
            fa3 += bias[4 * c + 3];
        }
        if (relu_flag) {
            fa0 = max(fa0, (float)0);
            fa1 = max(fa1, (float)0);
            fa2 = max(fa2, (float)0);
            fa3 = max(fa3, (float)0);
        }
        char4 res = make_char4(static_cast<char>(fa0),
                               static_cast<char>(fa1),
                               static_cast<char>(fa2),
                               static_cast<char>(fa3));
        char4* d = ((char4*)dout);
        d[index] = res;
    }
#endif
}

template <bool relu_flag>
SaberStatus saber_depthwise_conv_act_s8_s8(const void* input, void* output,
        int num, int cin, int hin, int win, int hout, int wout,
        int kw, int kh, int stride_w, int stride_h, int pad_w, int pad_h, float alpha,
        const void* weights, const float* bias, cudaStream_t stream) {

    CHECK_EQ(cin % 4, 0);
    int cin_4 = cin / 4;
    const int count = num * cin_4 * hout * wout;

    if (bias != nullptr) {
        depthwise_conv_1d_s8_s8<true, relu_flag><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
                count, input, num, cin_4, hin, win, hout, wout, kh,
                        kw, stride_h, stride_w, pad_h, pad_w,
                        output, weights, bias, alpha);
    } else {
        depthwise_conv_1d_s8_s8<false, relu_flag><<< CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>> (
                count, input, num, cin_4, hin, win, hout, wout, kh,
                        kw, stride_h, stride_w, pad_h,
                        pad_w, output, weights, nullptr, alpha);
    }
    return SaberSuccess;
}

template <bool bias_flag, bool relu_flag>
__global__ void depthwise_conv_1d_s8_f32(const int nthreads,
        const void* din, const int num, const int channels,
        const int hin, const int win, const int hout,
        const int wout, const int kernel_h, const int kernel_w,
        const int stride_h, const int stride_w, const int pad_h, const int pad_w,
        void* dout, const void* weight, const float* bias, float alpha = 1.f) {
#if __CUDA_ARCH__ > 600
    int size_channel_in = hin * win;
    int size_channel_out = hout * wout;
    int size_kernel = kernel_h * kernel_w;
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int pw = index % wout;
        const int ph = (index / wout) % hout;
        const int c = (index / size_channel_out) % channels;
        const int n = index / size_channel_out / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = hstart + kernel_h;
        int wend = wstart + kernel_w;

        int khstart = hstart < 0 ? 0 - hstart : 0;
        int kwstart = wstart < 0 ? 0 - wstart : 0;

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, hin);
        wend = min(wend, win);

        int aveval0 = 0;
        int aveval1 = 0;
        int aveval2 = 0;
        int aveval3 = 0;

        const int* bottom_slice = (const int*)din + (n * channels + c) * size_channel_in;
        const int* weight_slice = (const int*)weight + c * size_kernel;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int in_data = bottom_slice[h * win + w];
                int weight_data = weight_slice[(khstart + h - hstart) * kernel_w
                                               + (kwstart + w - wstart)];
                int mask_weight;
                mask_weight = MASK0 & weight_data;
                aveval0 = __dp4a(in_data, mask_weight, aveval0);
                mask_weight = MASK1 & weight_data;
                aveval1 = __dp4a(in_data, mask_weight, aveval1);
                mask_weight = MASK2 & weight_data;
                aveval2 = __dp4a(in_data, mask_weight, aveval2);
                mask_weight = MASK3 & weight_data;
                aveval3 = __dp4a(in_data, mask_weight, aveval3);
            }
        }
        float fa0 = static_cast<float>(aveval0);
        float fa1 = static_cast<float>(aveval1);
        float fa2 = static_cast<float>(aveval2);
        float fa3 = static_cast<float>(aveval3);
        fa0 *= alpha;
        fa1 *= alpha;
        fa2 *= alpha;
        fa3 *= alpha;

        if (bias_flag) {
            fa0 += bias[4 * c + 0];
            fa1 += bias[4 * c + 1];
            fa2 += bias[4 * c + 2];
            fa3 += bias[4 * c + 3];
        }
        if (relu_flag) {
            fa0 = max(fa0, (float)0);
            fa1 = max(fa1, (float)0);
            fa2 = max(fa2, (float)0);
            fa3 = max(fa3, (float)0);
        }

        int output_slice = hout * wout;
        int out_idx = (index % output_slice) + 4 * c * output_slice;
        ((float*)dout)[out_idx] = fa0; out_idx += output_slice;
        ((float*)dout)[out_idx] = fa1; out_idx += output_slice;
        ((float*)dout)[out_idx] = fa2; out_idx += output_slice;
        ((float*)dout)[out_idx] = fa3;
    }
#endif
}

template <bool relu_flag>
SaberStatus saber_depthwise_conv_act_s8_f32(const void* input, void* output,
        int num, int cin, int hin, int win, int hout, int wout,
        int kw, int kh, int stride_w, int stride_h, int pad_w, int pad_h, float alpha,
        const void* weights, const float* bias, cudaStream_t stream) {

    CHECK_EQ(cin % 4, 0);
    int cin_4 = cin / 4;
    const int count = num * cin_4 * hout * wout;

    if (bias != nullptr) {
        depthwise_conv_1d_s8_f32<true, relu_flag><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
                count, input, num, cin_4, hin, win, hout, wout, kh,
                        kw, stride_h, stride_w, pad_h, pad_w,
                        output, weights, bias, alpha);
    } else {
        depthwise_conv_1d_s8_f32<false, relu_flag><<< CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>> (
                count, input, num, cin_4, hin, win, hout, wout, kh,
                        kw, stride_h, stride_w, pad_h,
                        pad_w, output, weights, nullptr, alpha);
    }
    return SaberSuccess;
}

#define INSTANCE_CONVACT(ifrelu) \
template \
    SaberStatus saber_depthwise_conv_act<ifrelu> (const float* input, float* output, \
    int num, int cin, int hin, int win, int hout, int wout, \
    int kw, int kh, int stride_w, int stride_h, \
    int pad_h, int pad_w, const float* weights, const float* bias, cudaStream_t stream);

#define INSTANCE_CONVACT_S8_S8(ifrelu) \
template \
SaberStatus saber_depthwise_conv_act_s8_s8<ifrelu>(const void* input, void* output, \
        int num, int cin, int hin, int win, int hout, int wout, \
        int kw, int kh, int stride_w, int stride_h, int pad_w, int pad_h, float alpha, \
        const void* weights, const float* bias, cudaStream_t stream);

#define INSTANCE_CONVACT_S8_F32(ifrelu) \
template \
SaberStatus saber_depthwise_conv_act_s8_f32<ifrelu>(const void* input, void* output, \
        int num, int cin, int hin, int win, int hout, int wout, \
        int kw, int kh, int stride_w, int stride_h, int pad_w, int pad_h, float alpha, \
        const void* weights, const float* bias, cudaStream_t stream);

INSTANCE_CONVACT(true);
INSTANCE_CONVACT(false);
INSTANCE_CONVACT_S8_S8(true);
INSTANCE_CONVACT_S8_S8(false);
INSTANCE_CONVACT_S8_F32(true);
INSTANCE_CONVACT_S8_F32(false);

} //namespace anakin
} //namespace anakin
